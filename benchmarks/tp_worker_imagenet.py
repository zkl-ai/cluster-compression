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
from multiprocessing import Queue, get_context, Manager
from http.server import HTTPServer, BaseHTTPRequestHandler
import socket
import json
import requests
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
    
    # pruning parameters
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--method", type=str, default='l2')
    parser.add_argument("--global-pruning", default=False, action="store_true")
    parser.add_argument("--target-flops", type=float, default=2.0, help="GFLOPs of pruned model")
    parser.add_argument("--soft-keeping-ratio", type=float, default=0.0)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--delta_reg", type=float, default=1e-4)
    parser.add_argument("--max-pruning-ratio", default=1.0, type=float, help="maximum channel pruning ratio")
    parser.add_argument("--sl-epochs", type=int, default=None)
    parser.add_argument("--sl-resume", type=str, default=None)
    parser.add_argument("--sl-lr", default=None, type=float, help="learning rate")
    parser.add_argument("--sl-lr-step-size", default=None, type=int, help="milestones for learning rate decay")
    parser.add_argument("--sl-lr-warmup-epochs", default=None, type=int, help="the number of epochs to warmup (default: 0)")
    return parser


def runner(args, req, lock):
    device = 'cuda'
    model_type, idx, prune_ratios, callback_address = req
    print("Creating model")
    model = registry.get_model(num_classes=1000, name=model_type, pretrained=args.pretrained, target_dataset='imagenet') #torchvision.models.__dict__[args.model](pretrained=args.pretrained) #torchvision.models.get_model(args.model, weights=args.weights, num_classes=num_classes)
    model.eval()
    print("="*16)
    print(model)
    example_inputs = torch.randn(1, 3, 224, 224)
    base_ops, base_params = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    print("Params: {:.4f} M".format(base_params / 1e6))
    print("ops: {:.4f} G".format(base_ops / 1e9))
    print("="*16)
    
    print("Pruning model...")

    ignored_layers = []
    pruning_ratio_dict = {}
    pruning_ratio_idx = 0
    if isinstance(model, torchvision.models.resnet.ResNet):
        for m in model.modules():
            if isinstance(m, torchvision.models.resnet.Bottleneck): 
                pruning_ratio_dict[m] = prune_ratios[pruning_ratio_idx]
                pruning_ratio_idx += 1
            if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
                ignored_layers.append(m) # DO NOT prune the final classifier!

    elif isinstance(model, torchvision.models.vgg.VGG):
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
    
    imp = tp.importance.MagnitudeImportance(p=2)
    # print(pruning_ratio_dict)
    pruner = tp.pruner.MetaPruner(
            model,
            example_inputs,
            importance=imp,
            ch_sparsity=1.0,
            ch_sparsity_dict=pruning_ratio_dict,
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

    # Test forward in eval mode
    print("====== Forward (Inferece with torch.no_grad) ======")
    model = model.eval().to(device)
    example_inputs = torch.randn(args.batch_size, 3, 224, 224).to(device)
    with torch.no_grad():
        laterncy_mu, latency_std= tp.utils.benchmark.measure_latency(model, example_inputs, repeat=10)
        print('laterncy: {:.4f} +/- {:.4f} ms'.format(laterncy_mu, latency_std))

    data = {
        'idx': idx,
        'loss': 100.0,
        'mem': 0.0,
        'lat': laterncy_mu,
        'weights': 0.0,
        'macs': 0.0,
        'mmin': 0.0,
        'mavg': 0.0,
        'device':hostname+":"+ip_address,
    }

    print(data)

    # Release GPU memory
    del model
    # torch.cuda.empty_cache()
    # gc.collect()


    req = requests.post(callback_address, data=json.dumps(data))
    print(req)
    status_code = req.status_code
    if status_code == 200:
        print("sucessful")
    else:
        print("wrong request with response".format(status_code))

    lock.release()


def consumer(name, args, individual_queue):
    with Manager() as m:
        lock = m.Lock()
        while True:
            lock.acquire()
            req = individual_queue.get()
            ctx = get_context('spawn')
            # t = ctx.Process(target=runner, args=(req, test_data_mem_lat_iter, network_utils, lock))
            runner(args,req,lock)


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

    ctx = get_context('fork')

    server = (ip_address, args.listen_port)
    p1 = ctx.Process(target=producer, args=('producer', individual_queue, server))
    c1 = ctx.Process(target=consumer, args=('consumer', args, individual_queue))
    p1.start()
    c1.start()
    return
    
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    worker(args)