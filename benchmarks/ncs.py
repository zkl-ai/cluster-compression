import numpy as np
import logging
import os
import hashlib
import joblib
import registry
import torch_pruning as tp
import gc
import torch, torchvision
import json
import copy
import torch.distributed as dist
from HAMP import network_utils as networkUtils
import engine.models as models

def calBdistance(param1, param2, sigma1, sigma2):
    """计算分布之间的距离
    参数：
        param1 (np.ndarray): 分布1的均值
        sigma1 (np.ndarray): 分布1的协方差

        param2 (np.ndarray): 分布2的均值
        sigma2 (np.ndarray): 分布2的协方差

    返回值：
        分布之间的距离值
    """
    # 计算均值的差异
    xi_xj = param1 - param2
    
    # 计算方差
    big_sigma1 = np.square(sigma1)
    big_sigma2 = np.square(sigma2)
    big_sigma = (big_sigma1 + big_sigma2) / 2
    
    # 避免除以零
    small_value = 1e-8
    big_sigma += small_value
    big_sigma1 += small_value
    big_sigma2 += small_value

    # 计算部分1
    part1 = 0.125 * np.sum(np.square(xi_xj) / big_sigma)

    # 计算部分2
    part2 = np.sum(np.log(big_sigma)) - 0.5 * (np.sum(np.log(big_sigma1)) + np.sum(np.log(big_sigma2)))

    # 返回总距离
    return part1 + 0.5 * part2


class PruningRateOptimizer:
    def __init__(self, model_name, model, layer_sizes, base_layer_sizes, cur_prune_rate, seed, eval_acc, 
                 r=0.95, epoch=5,up_prune_rate=0.2,
                 population_size=10, num_generations=100, acc_threshold=0.9, max_attempts=100, log_dir='./logs/ea-logs'):
        # log & seed
        self.seed = seed
        np.random.seed(seed)
        self.log_dir = log_dir
        
        # 设置日志记录
        if dist.get_rank() == 0:
            self.setup_logging()

        # 初始化缓存
        self.fitness_cache = {} # weight sum of latency and accuracy
        self.fitness_cache_tuple = {} # (latency, accuracy)
        
        
        # model
        self.model_name = model_name
        self.model = model
        self.eval_acc = eval_acc
        self.layer_sizes = layer_sizes
        self.base_layer_sizes = base_layer_sizes
        
        
        # ea param
        self.population_size = population_size
        self.num_genes = len(self.layer_sizes)
        self.num_generations = num_generations
        self.r = r
        self.epoch = epoch
        self.successful_count = np.zeros(self.population_size)
        self.up_prune_rate = up_prune_rate

        self.acc_threshold = acc_threshold
        self.max_attempts = max_attempts
        



        # latency model 
        self.latency_models = []
        directory = f'surrogate/{model_name}_latency_model'
        for filename in os.listdir(directory):
            if filename.endswith('.pkl'):
                self.log(f'load {filename} latency model...')
                latency_model_path = os.path.join(directory, filename)
                self.latency_models.append(joblib.load(latency_model_path))
        
        self.cur_prune_rate = cur_prune_rate
        self.base_latency = np.mean([m.predict(np.zeros((1,self.num_genes)))for m in self.latency_models])
        self.base_accuracy = self.eval_acc(self.model)
        
        cur_latency = np.mean([m.predict(np.array(cur_prune_rate).reshape(1,-1))for m in self.latency_models])
        self.log(cur_latency/self.base_latency)
        
        self.best_individual_history = [[np.zeros((self.num_genes)).tolist(), cur_latency/self.base_latency, 1.0]] # prune_rate, latency, acc
        
        self.best_found = None
        self.best_value = float('inf')
        if 'vgg' in self.model_name:
            self.network_utils = networkUtils.__dict__['vgg19'](model, (3,224,224), '/data/workspace/datasets/',0.1)
        self.initialize_population()
        self.sigma = self.initialize_sigma()
    
        
        
    def setup_logging(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        logging.basicConfig(
            filename=os.path.join(self.log_dir, 'optimization.log'),
            filemode='w',
            format='%(asctime)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger()
        # 在程序启动时清空文件内容
        population_path = os.path.join(self.log_dir, 'population.txt')
        # 以写入模式打开文件，清空文件内容
        open(population_path, 'w').close()

    def log(self, message):
        # print(message)
        if dist.get_rank() == 0:
            self.logger.info(message)

    def hash_individual(self, individual):
        individual_kernels = (self.layer_sizes * individual).astype(int)
        return hashlib.sha256(individual_kernels.tobytes()).hexdigest()

    def initialize_population(self):
        population = np.zeros((self.population_size, self.num_genes))
        population_idx = 1
        while population_idx < self.population_size:
            individual = self.generate_individual()
            
            population[population_idx] = individual
            population_idx += 1
            
        self.population = population
        f_x = np.array([self.evaluate_fitness(self.population[i]) 
                for i in range(self.population_size)])
        
        if dist.get_rank() == 0:
            self.save_population(generation=0)
        

    def initialize_sigma(self):
        """
        初始化sigma矩阵
        :return: 初始化的sigma矩阵
        """
        # 使用广播功能创建 sigma 矩阵
        sigma = np.ones(self.num_genes) * 0.05
        
        # 创建一个 (population_size, num_genes) 的矩阵，其中每一行都是sigma
        sigma_matrix = np.tile(sigma, (self.population_size, 1))
        return sigma_matrix
    
    def generate_individual(self):
        individual = np.zeros(self.num_genes)
        selected_num_genes = int(self.num_genes)
        selected_indices = np.random.choice(self.num_genes, selected_num_genes, replace=False)
        for gene_idx in selected_indices:
            layer_size = self.layer_sizes[gene_idx]
            individual[gene_idx] = np.random.randint(0, layer_size // 10) / layer_size
        
        individual = np.clip(individual, 0.0, self.up_prune_rate)
        individual = np.minimum(individual, 1-1/self.layer_sizes)
        individual = (self.layer_sizes * individual).astype(int)
        individual = individual / self.layer_sizes
        return individual

    def prune_vgg(self, prune_rate):
        self.log("Creating model")
        model = copy.deepcopy(self.model).to('cpu')
        model.eval()
        self.log("="*16)
        example_inputs = torch.randn(1, 3, 224, 224).to('cpu')
        base_ops, base_params = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        
        self.log("Pruning model...")
        network_utils = self.network_utils

        # Get the original network_def
        network_def = network_utils.get_network_def_from_model(model)

        # Generate simplified_network_def
        simplified_network_def = copy.deepcopy(network_def)
        for var_idx in range(len(prune_rate)):
            simplified_network_def = (
                network_utils.simplify_network_def_based_on_variable(simplified_network_def, var_idx,
                                                                        1.0-prune_rate[var_idx]))
        # Generate simplified_model
        model = network_utils.simplify_model_based_on_network_def(simplified_network_def, model)

        
        model = model.to('cpu')
        self.log("="*16)
        self.log("After pruning:")
        # self.log(model)
        pruned_ops, pruned_size = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        self.log("Params: {:.2f} M => {:.2f} M ({:.2f}%)".format(base_params / 1e6, pruned_size / 1e6, pruned_size / base_params * 100))
        self.log("Ops: {:.2f} G => {:.2f} G ({:.2f}%, {:.2f}X )".format(base_ops / 1e9, pruned_ops / 1e9, pruned_ops / base_ops * 100, base_ops / pruned_ops))
        self.log("="*16)
        del simplified_network_def, network_def
        return model
        
    
    def prune(self, prune_rate):
        self.log("Creating model")
        model = copy.deepcopy(self.model).to('cpu')
        model.eval()
        self.log("="*16)
        example_inputs = torch.randn(1, 3, 224, 224).to('cpu')
        base_ops, base_params = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        
        self.log("Pruning model...")
        ignored_layers = []
        pruning_ratio_dict = {}
        pruning_ratio_idx = 0
        if isinstance(model, torchvision.models.resnet.ResNet):
            for m in model.modules():
                if isinstance(m, torchvision.models.resnet.Bottleneck): 
                    pruning_ratio_dict[m] = prune_rate[pruning_ratio_idx]
                    pruning_ratio_idx += 1
                if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
                    ignored_layers.append(m) # DO NOT prune the final classifier!

        elif isinstance(model, torchvision.models.vgg.VGG):
            for m in model.modules():
                if isinstance(m, torch.nn.Conv2d):
                    pruning_ratio_dict[m] = prune_rate[pruning_ratio_idx]
                    pruning_ratio_idx += 1
                if isinstance(m, torch.nn.Linear):
                    if m.out_features == 1000:
                        ignored_layers.append(m) # DO NOT prune the final classifier!
                    else:
                        pruning_ratio_dict[m] = prune_rate[pruning_ratio_idx]
                        pruning_ratio_idx += 1
        elif isinstance(model, models.imagenet.MobileNetV1):
            for dep_conv in model.features:
                pruning_ratio_dict[dep_conv] = prune_rate[pruning_ratio_idx]
                pruning_ratio_idx += 1
            ignored_layers.append(model.classifier) # DO NOT prune the final classifier!
            

        imp = tp.importance.MagnitudeImportance(p=2)
        pruner = tp.pruner.MetaPruner(
            model,
            example_inputs,
            importance=imp,
            pruning_ratio=1.0,
            pruning_ratio_dict=pruning_ratio_dict,
            ignored_layers=ignored_layers,
        )
        model = model.to('cpu')
        self.log("="*16)
        pruner.step()
        self.log("After pruning:")
        # self.log(model)
        pruned_ops, pruned_size = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        self.log("Params: {:.2f} M => {:.2f} M ({:.2f}%)".format(base_params / 1e6, pruned_size / 1e6, pruned_size / base_params * 100))
        self.log("Ops: {:.2f} G => {:.2f} G ({:.2f}%, {:.2f}X )".format(base_ops / 1e9, pruned_ops / 1e9, pruned_ops / base_ops * 100, base_ops / pruned_ops))
        self.log("="*16)
        return model

    def evaluate_fitness(self, prune_rate):
        individual_hash = self.hash_individual(prune_rate)

        # 如果个体已经被评估过，则返回缓存的适应度
        if individual_hash in self.fitness_cache:
            return self.fitness_cache[individual_hash]

        # 计算延迟
        latency_pr = ((prune_rate * self.layer_sizes).astype(dtype=np.int32) + (self.cur_prune_rate * self.base_layer_sizes).astype(dtype=np.int32)) / self.base_layer_sizes
        latency = np.mean([m.predict(np.array(latency_pr).reshape(1, -1)) for m in self.latency_models])
        # 计算准确率（这里假设有一个函数来计算准确率）
        prune_model = None
        if 'vgg' in self.model_name:
            prune_model = self.prune_vgg(prune_rate)
        else:
            prune_model = self.prune(prune_rate)
        


        accuracy = self.eval_acc(prune_model)
        
        # 归一化延迟
        normalized_latency = latency / self.base_latency
        normalized_accuracy = accuracy / self.base_accuracy
        self.log(f'latency_pr:{latency}, base_latency:{self.base_latency}, percentage: {normalized_latency}')

        # 综合适应度计算
        
        fitness = normalized_latency 
        if normalized_accuracy <  self.acc_threshold :
            # fitness = fitness + (1-normalized_accuracy) / self.acc_threshold
            fitness = (1-normalized_accuracy) / (1-self.acc_threshold)
        
        # 将适应度存储在缓存中
        self.fitness_cache[individual_hash] = fitness
        self.fitness_cache_tuple[individual_hash] = (normalized_latency, normalized_accuracy, fitness)
        # 更新最好个体
        best_individual, best_lat, best_acc = self.best_individual_history[-1][0], self.best_individual_history[-1][1], self.best_individual_history[-1][2]
        if normalized_accuracy >= self.acc_threshold and normalized_latency < best_lat:
            self.best_individual_history.append([prune_rate.tolist(), normalized_latency, normalized_accuracy])
        
        self.log(f"Latency & Accuracy & Fitness: {json.dumps(self.fitness_cache_tuple[individual_hash])}\n")
        del prune_model
        gc.collect()
        return fitness


    def gaussian_mutation(self, x, sigma):
        """
        高斯变异操作
        :param x: 当前解
        :param sigma: 当前步长
        :return: 变异后的新解
        """
        x_p = None
        attempts = 0
        while attempts < self.max_attempts:
            attempts += 1
            
            x_prime = np.clip(x + sigma * np.random.randn(self.num_genes), 0.0, self.up_prune_rate)
            x_prime = np.minimum(x_prime, 1-1/self.layer_sizes)
            x_prime = (self.layer_sizes * x_prime).astype(int)
            x_prime = x_prime / self.layer_sizes
            x_prime_hash = self.hash_individual(x_prime)
            if x_prime_hash not in self.fitness_cache:
                x_p = x_prime
        if x_p is None:
            x_p = self.generate_individual()
        return x_prime
    
    def update_sigma(self, i):
        """
        根据1/5成功规则更新步长
        :param successful_count: 替换成功的次数
        """
        if self.successful_count[i]  > 0.2 * self.epoch:
            self.sigma[i] /= self.r
        elif self.successful_count[i] < 0.2 * self.epoch:
            self.sigma[i] *= self.r
        self.successful_count[i] = 0

    def save_population(self, generation):
        population_path = os.path.join(self.log_dir, 'population.txt')
        # 确保目录存在
        os.makedirs(os.path.dirname(population_path), exist_ok=True)
        
        # 打开文件以追加模式写入
        with open(population_path, 'a') as file:
            # 写入 generation 信息
            file.write(f"Generation: {generation}\n")
            file.write("=" * 50 + "\n")
            
            # 写入种群中的每个个体及其对应的延迟和精度
            for row in self.population:
                # 转换数组数据为字符串
                pruning_rate_str = json.dumps(row.tolist())
                # 获取延迟和精度数据
                latency_acc = self.fitness_cache_tuple[self.hash_individual(row)]
                latency_acc_str = json.dumps(latency_acc)
                
                # 写入修剪率
                file.write(f"Pruning Rate: {pruning_rate_str}\n")
                # 写入延迟和精度
                file.write(f"Latency & Accuracy & Fitness: {latency_acc_str}\n")
                file.write("-" * 50 + "\n")
        
        # 记录日志信息
        self.log(f"Population for generation {generation} saved to {population_path}")


    def optimize(self):
        latency, accuracy = np.inf, np.inf
        generation = 0

        # 用于存储每代的延迟和精度下降数据
        history = {
            "generations": [],
            "normalized_latency": [],
            "accuracy": []
        }
        best_ind_path = os.path.join(self.log_dir, 'best_idv.json')

        for t in range(1,self.num_generations+1):
            # 生成 λ_t 的值
            # lambda_t = np.random.normal(1, 0.1 - 0.1 * t / self.num_generations)
            lambda_t = np.random.randn() * (0.1-0.1*t/(self.num_generations/self.population_size)) + 1.0
            
            # 生成所有子代
            x_prime = np.array([self.gaussian_mutation(self.population[i], self.sigma[i]) 
                                for i in range(self.population_size)])
            
            # 计算所有子代和当前解的适应度
            f_x_prime = np.array([self.evaluate_fitness(x_prime[i]) 
                                for i in range(self.population_size)])
            f_x = np.array([self.evaluate_fitness(self.population[i]) 
                            for i in range(self.population_size)])
            
            # 计算相关性
            corr_x = np.zeros(self.population_size)
            corr_x_prime = np.zeros(self.population_size)
            
            for i in range(self.population_size):
                distances_parent = np.array([calBdistance(self.population[i], self.population[j], self.sigma[i], self.sigma[j])
                                    for j in range(self.population_size) if j != i])
                corr_x[i] = distances_parent.min()
                distances_offspring = np.array([calBdistance(x_prime[i], x_prime[j], self.sigma[i], self.sigma[j])
                                    for j in range(self.population_size) if j != i])
                corr_x_prime[i] = distances_offspring.min()

            # 归一化 f 和 Corr
            f_sum = f_x + f_x_prime
            corr_sum = corr_x + corr_x_prime
            
            # 防止除以零的情况
            f_sum[f_sum == 0] = 1e-10
            corr_sum[corr_sum == 0] = 1e-10
            
            norm_f_x_prime = f_x_prime / f_sum
            norm_corr_x_prime = corr_x_prime / corr_sum
            
            
            # 决定是否替换当前解
            for i in range(self.population_size):
                self.log(f"individual {i}, norm_f_x_p: {norm_f_x_prime[i]}, norm_corr_x_p: {norm_corr_x_prime[i]}, {norm_f_x_prime[i] / norm_corr_x_prime[i]}, lambda_t: {lambda_t}")
                if norm_f_x_prime[i] / norm_corr_x_prime[i] < lambda_t:
                    self.population[i] = x_prime[i]
                    self.successful_count[i] += 1
                    
                # 更新最佳解
                if self.best_found is None or f_x_prime[i] < self.best_value:
                    self.best_found = x_prime[i]
                    self.best_value = f_x_prime[i]
            
            # 更新步长
            if (t + 1) % self.epoch == 0:
                for i in range(self.population_size):
                    self.update_sigma(i)
            # 记录当前代的延迟和精度下降
            history["generations"].append(generation)
            history["normalized_latency"].append(latency)
            history["accuracy"].append(accuracy)
            
            if dist.get_rank() == 0:
                self.save_population(t)

            self.log(f"Generation {t}: Best Fitness = {self.best_value}, Best Individual = {json.dumps(self.best_found.tolist())}")
            with open(best_ind_path, 'w') as f:
                json.dump(self.best_individual_history, f, indent=4)
            self.log(f"Best individual history saved to {best_ind_path}.")
            

        # 保存延迟和精度下降的变化数据
        history_path = os.path.join(self.log_dir, 'history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        self.log(f"Latency and accuracy history saved to {history_path}")

        return self.best_individual_history[-1][0]

