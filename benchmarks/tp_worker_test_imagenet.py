import datetime
import os, sys
import time
import warnings
import registry
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

from engine.utils.imagenet_utils import presets, transforms, utils, sampler
import engine.models as models
import torchvision
import torch
import torch.utils.data
import torchvision
#from sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode

import torch_pruning as tp 
from functools import partial
# from multiprocessing import Queue, get_context, Manager
from http.server import HTTPServer, BaseHTTPRequestHandler
import socket
import json
import requests
import gc
import threading
import psutil
from queue import Queue
def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--data-path", default="~/Datasets/ImageNet/", type=str, help="dataset path")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--pretrained", action="store_true")
    
    

    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)")
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, metavar="W", help="weight decay (default: 1e-4)", dest="weight_decay")
    parser.add_argument("--norm-weight-decay", default=None, type=float, help="weight decay for Normalization layers (default: None, same value as --wd)")
    parser.add_argument("--bias-weight-decay", default=None, type=float, help="weight decay for bias parameters of all layers (default: None, same value as --wd)")
    parser.add_argument("--transformer-embedding-decay", default=None, type=float, help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)")
    parser.add_argument("--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing")
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)")
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=100, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--cache-dataset", dest="cache_dataset", help="Cache the datasets for quicker initialization. It also serializes the transforms", action="store_true")
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", action="store_true")
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters")
    parser.add_argument("--model-ema-steps", type=int, default=32, help="the number of iterations that controls how often to update the EMA model (default: 32)")
    parser.add_argument("--model-ema-decay", type=float, default=0.99998, help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)")
    parser.add_argument("--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only.")
    parser.add_argument("--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)")
    parser.add_argument("--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)")
    parser.add_argument("--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)")
    parser.add_argument("--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)")
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument("--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    
    
    parser.add_argument('--listen_port', type=int, default=8083, help='listen port which is used to receive individual from server')
    
    return parser


def runner(args, req):
    device = 'cuda'
    model_type, idx, prune_ratios, callback_address = req
    print("Creating model")
    # metapruning
    # script_dir = os.getcwd()

    # root_dir = os.path.abspath(os.path.join(script_dir, 'metapruning/resnet50/1G'))
    # # 将根目录添加到 sys.path，这样可以导入根目录中的模块
    # sys.path.append(root_dir)

    # # 导入模型定义
    # from resnet import resnet50  # 替换为模型类的实际名称

    # # 初始化模型
    # model = resnet50()
    
    # autoslim
    # autoslim
    script_dir = os.getcwd()

    root_dir = os.path.abspath(os.path.join(script_dir, 'slimmable_networks'))
    # 将根目录添加到 sys.path，这样可以导入根目录中的模块
    sys.path.append(root_dir)

    from models.autoslim_resnet import Model
    from utils.model_profiling import model_profiling
    from models.slimmable_ops import FLAGS


    FLAGS.width_mult_list = [1,2,3]
    FLAGS.channel_num_list = [
        [24, 72, 8, 16, 72, 16, 16, 72, 8, 8, 72, 240, 16, 24, 240, 16, 24, 240, 16, 16, 240, 16, 24, 240, 704, 72, 72, 704, 48, 72, 704, 48, 72, 704, 72, 72, 704, 48, 48, 704, 48, 48, 704, 1680, 280, 240, 1680, 120, 200, 1680, 280, 280, 1680, 1000],  # 570M
        [32, 72, 16, 16, 72, 16, 16, 72, 8, 8, 72, 280, 24, 40, 280, 16, 24, 280, 16, 24, 280, 24, 40, 280, 880, 72, 72, 880, 72, 96, 880, 72, 72, 880, 72, 72, 880, 48, 72, 880, 168, 96, 880, 2016, 360, 280, 2016, 320, 360, 2016, 480, 440, 2016, 1000],  # 1000M
        [40, 120, 16, 16, 120, 24, 24, 120, 16, 24, 120, 440, 48, 72, 440, 32, 48, 440, 40, 56, 440, 64, 72, 440, 1024, 168, 168, 1024, 96, 144, 1024, 120, 168, 1024, 120, 144, 1024, 120, 144, 1024, 256, 256, 1024, 2016, 512, 512, 2016, 480, 480, 2016, 512, 512, 2016, 1000],  # 2006M
        # [48, 192, 16, 16, 192, 56, 56, 192, 24, 40, 192, 512, 88, 112, 512, 64, 104, 512, 104, 112, 512, 80, 128, 512, 1024, 240, 216, 1024, 144, 216, 1024, 192, 192, 1024, 144, 240, 1024, 240, 256, 1024, 256, 256, 1024, 2016, 512, 512, 2016, 480, 512, 2016, 512, 512, 2016, 1000],  # 3005M
    ]
    model = Model()
    
    # model = registry.get_model(num_classes=10, name=model_type, pretrained=args.pretrained, target_dataset='mnist')
    model.eval()

    example_inputs = torch.randn(1, 3, 224, 224)
    
    print("Pruning model...")
    ignored_layers = []
    pruning_ratio_dict = {}
    pruning_ratio_idx = 0
    if isinstance(model, Model):
        example_inputs = torch.randn(1, 3, 224, 224)
        for m in model.modules():
            if isinstance(m, torchvision.models.resnet.Bottleneck): 
                pruning_ratio_dict[m] = prune_ratios[pruning_ratio_idx]
                pruning_ratio_idx += 1
            if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
                ignored_layers.append(m) # DO NOT prune the final classifier!

    elif isinstance(model, torchvision.models.vgg.VGG):
        example_inputs = torch.randn(1, 3, 224, 224)
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                pruning_ratio_dict[m] = prune_ratios[pruning_ratio_idx]
                pruning_ratio_idx += 1
            if isinstance(m, torch.nn.Linear):
                if m.out_features == 1000:
                    ignored_layers.append(m) # DO NOT prune the final classifier!
                else:
                    pruning_ratio_dict[m] = prune_ratios[pruning_ratio_idx]
                    pruning_ratio_idx += 1
    elif isinstance(model, registry.MNIST_MODEL_DICT['lenet5']):
        example_inputs = torch.randn(1, 1, 28, 28)
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                pruning_ratio_dict[m] = prune_ratios[pruning_ratio_idx]
                pruning_ratio_idx += 1
            if isinstance(m, torch.nn.Linear):
                if m.out_features == 10:
                    ignored_layers.append(m) # DO NOT prune the final classifier!
                else:
                    pruning_ratio_dict[m] = prune_ratios[pruning_ratio_idx]
                    pruning_ratio_idx += 1
        

    print("="*16)
    print(model)
    
    base_ops, base_params = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    print("Params: {:.4f} M".format(base_params / 1e6))
    print("ops: {:.4f} G".format(base_ops / 1e9))
    print("="*16)

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
    print("="*16)
    print("After pruning:")
    pruner.step()
    print(model)
    pruned_ops, pruned_size = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    print("Params: {:.2f} M => {:.2f} M ({:.2f}%)".format(base_params / 1e6, pruned_size / 1e6, pruned_size / base_params * 100))
    print("Ops: {:.2f} G => {:.2f} G ({:.2f}%, {:.2f}X )".format(base_ops / 1e9, pruned_ops / 1e9, pruned_ops / base_ops * 100, base_ops / pruned_ops))
    print("="*16)

    print("====== Forward (Inference with torch.no_grad) ======")
    model = model.eval().to(device)
    batch_example_inputs = example_inputs.repeat(args.batch_size, 1, 1, 1).to(device)
    with torch.no_grad():
        latency_mu, latency_std = tp.utils.benchmark.measure_latency(model, batch_example_inputs, repeat=10)
        print('latency: {:.4f} +/- {:.4f} ms'.format(latency_mu, latency_std))

    data = {
        'idx': idx,
        'loss': 100.0,
        'mem': psutil.virtual_memory().percent,
        'lat': latency_mu,
        'weights': 0.0,
        'macs': 0.0,
        'mmin': 0.0,
        'mavg': 0.0,
        'device': f'{hostname}:{ip_address}',
    }

    print(data)

    del model, example_inputs, batch_example_inputs
    torch.cuda.empty_cache()
    gc.collect()

    req = requests.post(callback_address, data=json.dumps(data))
    print(req)
    if req.status_code == 200:
        print("successful")
    else:
        print(f"wrong request with response {req.status_code}")



def consumer(name, args, individual_queue):
    while True:
        req = individual_queue.get()
        runner(args, req)


def producer(name, individual_queue, server=('localhost', 8080)):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            post_data = self.rfile.read(int(self.headers['content-length']))
            msg = json.loads(post_data.decode())
            print("=================================")
            print(post_data.decode())
            print("=================================")
            try:
                variables = msg['pop']
                model = msg['model']
                callback_address = msg['callback_address']
                idx = msg['idx']
                individual_queue.put((model, idx, variables, callback_address))
                self.response_data = {
                    'msg': "result received.",
                    "status_code": 200
                }
                self.response_code = 200
            except Exception:
                self.response_data = {
                    'msg': "result reject. wrong format result.",
                    "status_code": 401
                }
                self.response_code = 401
            self.send_response(self.response_code)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(self.response_data).encode('utf-8'))

    server = HTTPServer(server, Handler)
    server.serve_forever()
    
def get_host_ip():
    """
    get the host ip
    :return: ip, hostnames
    """
    try:
        hostname = socket.gethostname()
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip_address = s.getsockname()[0]
    finally:
        s.close()
        return ip_address, hostname
    

ip_address, hostname = get_host_ip()

def worker(args):
    print('worker starting ....')
    individual_queue = Queue()

    # Create server and consumer threads
    server_thread = threading.Thread(target=producer, args=('producer', individual_queue, (ip_address, args.listen_port)))
    consumer_thread = threading.Thread(target=consumer, args=('consumer', args, individual_queue))

    server_thread.start()
    consumer_thread.start()

    server_thread.join()
    consumer_thread.join()
    return
    
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    worker(args)