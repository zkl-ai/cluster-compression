from send_test import load_records
from tp_worker_imagenet import get_host_ip
import socket
import registry
import os
import gc
import pickle
from engine.utils.imagenet_utils import presets, transforms, utils, sampler
import engine.models as models
import torchvision
import torch
import torch.utils.data
import torchvision
import torch_pruning as tp 
ip_address, hostname = get_host_ip()

def split_dict(pruning_ratio_dict):
    total_items = len(pruning_ratio_dict)
    half_items = total_items // 2

    dict1 = {}
    dict2 = {}

    for i, (key, value) in enumerate(pruning_ratio_dict.items()):
        if i < half_items:
            dict1[key] = value
        else:
            dict2[key] = value
    
    return dict1, dict2


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--data-path", type=str, help="dataset path")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--root-dir", default="/data", type=str, help="path to save outputs")
    parser.add_argument("--output-dir", default="output_dir", type=str, help="path to save outputs")
    parser.add_argument("--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    
    return parser


def runner_imagenet(model_type, prune_ratios):
    device = 'cuda'
    print("Creating model")
    model = registry.get_model(num_classes=1000, name=model_type, pretrained=args.pretrained, target_dataset='imagenet')
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

    
    pruning_ratio_dict1, pruning_ratio_dict2 = split_dict(pruning_ratio_dict)
    imp = tp.importance.MagnitudeImportance(p=2)
    pruner1 = tp.pruner.MetaPruner(
        model,
        example_inputs,
        importance=imp,
        pruning_ratio=1.0,
        pruning_ratio_dict=pruning_ratio_dict1,
        ignored_layers=ignored_layers,
    )
    model = model.to('cpu')
    print("="*16)
    print("After pruning 1:")
    pruner1.step()
    print(model)
    
    pruner2 = tp.pruner.MetaPruner(
        model,
        example_inputs,
        importance=imp,
        pruning_ratio=1.0,
        pruning_ratio_dict=pruning_ratio_dict2,
        ignored_layers=ignored_layers,
    )
    model = model.to('cpu')
    print("="*16)
    print("After pruning 2:")
    pruner2.step()
    print(model)
    
    pruned_ops, pruned_size = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    print("Params: {:.2f} M => {:.2f} M ({:.2f}%)".format(base_params / 1e6, pruned_size / 1e6, pruned_size / base_params * 100))
    print("Ops: {:.2f} G => {:.2f} G ({:.2f}%, {:.2f}X )".format(base_ops / 1e9, pruned_ops / 1e9, pruned_ops / base_ops * 100, base_ops / pruned_ops))
    print("="*16)

    print("====== Forward (Inference with torch.no_grad) ======")
    model = model.eval().to(device)
    batch_example_inputs = torch.randn(args.batch_size, 3, 224, 224).to(device)
    with torch.no_grad():
        # latency_mu, latency_std = tp.utils.benchmark.measure_latency(model, batch_example_inputs, repeat=10)
        latency_mu, latency_std = 0, 0
        print('latency: {:.4f} +/- {:.4f} ms'.format(latency_mu, latency_std))
    
    del model, example_inputs, batch_example_inputs
    torch.cuda.empty_cache()
    gc.collect()
    
    return latency_mu

def main(args):
    output_path = f'{args.output_dir}/{hostname}-{args.data_path}' 
    log_path = f'{args.output_dir}/{hostname}-logs.txt'
    input_path = f'{args.root_dir}/{args.data_path}'
    records = load_records(input_path)
    for i, r in enumerate(records):
        prune_ratios = r.ratio
        latency = runner_imagenet(args.model, prune_ratios)
        r.latency[ip_address] = latency
        with open(output_path, "wb") as output_f:
            pickle.dump(records, output_f)
        with open(log_path, 'a') as log_f:
            log_f.write(f"[0:{i+1}) records are successfully tested!\n")



if __name__ == "__main__":
    args = get_args_parser().parse_args()
    output_dir = args.root_dir + '/' + hostname + '-' + args.output_dir
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(output_dir)
    args.output_dir = output_dir
    main(args)