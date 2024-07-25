import numpy as np
import csv
import random
import copy
import json
import os
import requests
from http.server import HTTPServer, BaseHTTPRequestHandler
from multiprocessing import get_context, Manager
from argparse import ArgumentParser
from record import Record
import pickle

def subnet_alexnet(model_template):
    # subnet = np.random.rand(1, model_template)
    subnet = []
    subnet_ratio = []
    for l in model_template:
        r = np.random.randint(1, l+1)
        subnet.append(r)
        subnet_ratio.append(r * 1.0 / l)
    subnet = np.array(subnet)
    subnet_ratio = np.array(subnet_ratio)
    return subnet_ratio

def subnet_vgg19(model_template):
    subnet = subnet_alexnet(model_template)
    # print(subnet)
    # print(subnet.shape)
    return subnet

def subnet_resnet50(model_template):
    subnet = subnet_alexnet(model_template)
    return subnet

def subnet_resnet34(model_template):
    target_idx = [1, 3, 5, 10, 12, 14, 19, 21, 23, 25, 27, 32, 34]
    subnet = []
    subnet_ratio = []
    for i, l in enumerate(model_template):
        if i in  target_idx:
            r = np.random.randint(1, l+1)
        else:
            r = l 
        subnet.append(r)
        subnet_ratio.append(r * 1.0 / l)
    subnet = np.array(subnet)
    subnet_ratio = np.array(subnet_ratio)
    # print(len(subnet_ratio))
    # print(subnet_ratio)
    return subnet_ratio

def subnet_parameter_gen(subnet_ratio, original_model):
    subnet_parameter = np.multiply(subnet_ratio, original_model)
    subnet_parameter = np.ceil(subnet_parameter).astype(dtype=np.int32) # ceil is used in hamp
    # print(subnet_parameter, subnet_parameter.shape, subnet_parameter.dtype)
    return subnet_parameter

def is_duplicate(subnet_parameter, subnets_parameter):
    for s in subnets_parameter:
        if (s==subnet_parameter).all():
            return True
    return False


def generate_subnet(n_sample, original_model, model_type):
    if model_type == 'alexnet':
        subnet_ratio_gen = subnet_alexnet
    elif model_type == 'vgg19':
        subnet_ratio_gen = subnet_vgg19
    elif model_type == 'vgg16':
        subnet_ratio_gen = subnet_vgg19
    elif model_type == 'resnet34':
        subnet_ratio_gen = subnet_resnet34
    elif model_type == 'resnet50':
        subnet_ratio_gen = subnet_resnet50
    else:
        raise Exception('wrong model type')
    # print('from generate subnet')
    # a = subnet_sample()
    # print(a.shape)
    subnets_ratio = []
    subnets_parameter = []
    while len(subnets_ratio) < n_sample:
        print(len(subnets_ratio))
        # generate subnet
        subnet_ratio = subnet_ratio_gen(original_model)
        subnet_parameter = subnet_parameter_gen(subnet_ratio, original_model)
        # remove duplicates
        while is_duplicate(subnet_parameter, subnets_parameter):
            subnet_ratio = subnet_ratio_gen()
            subnet_parameter = subnet_parameter_gen(subnet_ratio, original_model)
        # store subnet
        subnets_ratio.append(subnet_ratio)
        subnets_parameter.append(subnet_parameter)
    return subnets_ratio

def load_subnets(file_path, original_model):
    with open(file_path, 'rb') as f:
        records = pickle.load(f)
    subnets_ratio = []
    subnets_parameter = []
    for r in records:
        ratio = r.ratio 
        parameter = subnet_parameter_gen(ratio, original_model)
        subnets_ratio.append(ratio)
        subnets_parameter.append(parameter)
    return subnets_ratio, subnets_parameter



def generate_fake_subnet(file_path, n_sample, original_model, model_type):
    if model_type == 'alexnet':
        subnet_ratio_gen = subnet_alexnet
    elif model_type == 'vgg19':
        subnet_ratio_gen = subnet_vgg19
    elif model_type == 'resnet34':
        subnet_ratio_gen = subnet_resnet34
    else:
        raise Exception('wrong model type')
    # print('from generate subnet')
    # a = subnet_sample()
    # print(a.shape)
    subnets_ratio, subnets_parameter = load_subnets(file_path, original_model)
    subnets_ratio_return = []
    while len(subnets_ratio_return) < n_sample:
        print(len(subnets_ratio))
        # generate subnet
        subnet_ratio = subnet_ratio_gen(original_model)
        subnet_parameter = subnet_parameter_gen(subnet_ratio, original_model)
        # remove duplicates
        while is_duplicate(subnet_parameter, subnets_parameter):
            subnet_ratio = subnet_ratio_gen()
            subnet_parameter = subnet_parameter_gen(subnet_ratio, original_model)
        # store subnet
        subnets_ratio.append(subnet_ratio)
        subnets_parameter.append(subnet_parameter)
        subnets_ratio_return.append(subnet_ratio)
    return subnets_ratio_return


def generate_records(subnets, n_measure, file_path):
    records = []
    for subnet in subnets:
        record = Record(subnet)
        records.append(record)
    with open(file_path, "wb") as f:
        pickle.dump(records, f)
    

def save_subnet(subnets, n_measure, model_type):
    fields = ['model','device type', 'device id', ]
    latency_fields = []
    memory_fields = []
    for i in range(1, n_measure+1):
        l_field = "lantency %d" % i
        m_field = "memory %d" % i
        latency_fields.append(l_field)
        memory_fields.append(m_field)
    fields.extend(latency_fields)
    fields.extend(memory_fields)
    # print(len(fields))
    # print(fields)
    
    file_name = model_type+'_memtric.csv'
    # print(file_name)
    with open(file_name, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(fields)
        csv_writer.writerows(subnets)


if __name__=='__main__':
    n_sample = 5000 # number of subnets
    n_measure = 10 # number of measures for a subnet
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-m','--model_type', type=str, default='alexnet')
    arg_parser.add_argument('-f','--fake')
    args = arg_parser.parse_args()
    if args.model_type == 'alexnet':
        model = np.array([64,192,384,256,256,4096,4096]).astype(dtype=np.int32)
    elif args.model_type == 'vgg19':
        model = np.array([64, 64, 
                          128, 128, 
                          256, 256, 256, 256,
                          512, 512, 512, 512, 
                          512, 512, 512, 512,
                          4096, 4096,
                          ]).astype(dtype=np.int32)
    elif args.model_type == 'vgg16':
        model = np.array([64, 64,
                        128, 128, 
                        256, 256, 256, 
                        512, 512, 512, 
                        512, 512, 512, 
                        4096, 4096,
                        ]).astype(dtype=np.int32)
    elif args.model_type == 'resnet34':
        model = np.array([
            64,
            64, 64, 64, 64, 64, 64,
            128, 128, 128, 128, 128, 128, 128, 128, 128,
            256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
            512, 512, 512, 512, 512, 512, 512,
        ]) #[1, 3, 5, 10, 12, 14, 19, 21, 23, 25, 27, 32, 34]
    elif args.model_type == 'resnet50':
        model = np.array([
            64, 64, 64,
            128, 128, 128, 128,
            256, 256, 256, 256, 256, 256, 
            512, 512, 512,
        ])
    if args.fake:
        print("generate_fake_subnet...")
        file_path = "{}-{}-{}.pkl".format(args.model_type, 5000, n_measure)
        subnets = generate_fake_subnet(file_path, n_sample, model, args.model_type)
    else:
        print("generate_subnet...")
        file_path = "{}-{}-{}.pkl".format(args.model_type, n_sample, n_measure)
        subnets = generate_subnet(n_sample, model, args.model_type)
    subnets = np.array(subnets)
    print(subnets.shape)
    generate_records(subnets, n_measure, file_path=file_path)
    
