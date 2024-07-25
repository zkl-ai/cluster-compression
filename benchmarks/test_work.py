from argparse import ArgumentParser
import os
import pickle
import time
import torch
from shutil import copyfile
import subprocess
import sys
import warnings

import numpy as np
from collections import defaultdict
import re
import itertools
import socket
import json
import requests
from http.server import HTTPServer, BaseHTTPRequestHandler
# python3 HAMP_modify.py -w ./test_work  -mi 1 -test test -lp ./test -cp J1 -sp D3 -im ./models/helloworld/model_0.pth.tar

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
        return ip_address
    

def population_evaluation(args=None, population=None, iteration=None, num_variable=None):


    # population = np.ones((1,15))
    # for i in range(15):
    #     population[0][i] = 0.05 * (i+1)

    server = "http://10.16.61.187:8084"
    fn = ["LAT", "MEM"]
    callback_address = 'http://172.18.36.106:8888'
    model = "vgg16"
    timeout = 10
    data = {
        "pop" : population.tolist(),
        "model" : model,
        'callback_address':callback_address,
        "idx": 1,
    }
    num_variable = 1
    # population = np.ones((2,6))
    loss_index = num_variable
    memory_index = num_variable+1
    latency_index = num_variable+2
    weights_index = num_variable+3
    macs_index = num_variable+4

    KEEP_RUNING = [True]
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            post_data = self.rfile.read(int(self.headers['content-length']))
            msg = json.loads(post_data.decode())
            print(json.loads(post_data.decode()))
            try:
                metrics = msg
                idx = metrics['idx']
                population[idx][latency_index] = metrics["lat"]
                population[idx][memory_index] = metrics["mem"]
                self.response_data =  {
                                'msg' : "result received. wrong format result.",
                                "status_code":200
                                }
                self.response_code = 200
                keep_running(False)
            except Exception as e :
                self.response_data =  {
                                'msg' : "result received. wrong format result.",
                                "status_code":200
                                }
                self.response_code = 401
                print(e)
            self.send_response(self.response_code)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(self.response_data).encode('utf-8'))

    def keep_running(flag=None):
        if flag is not None:
            KEEP_RUNING[0] = flag
        return KEEP_RUNING[0]

    req = requests.post(server, data=json.dumps(data), timeout=timeout)
    status_code = req.status_code
    msg = json.loads(req.text)
    print(msg)
    # 异步，修改为监听一个端口号
    pop = np.ones((2,10))
    if status_code == 200:
        host = ('172.18.36.106', 8888)
        server = HTTPServer(host, Handler)
        
        while keep_running():
            server.handle_request()
    else:
        print("wrong request")

if __name__ == "__main__":
    import time 
    start = time.time()
    population = [
        # np.array([0.265625,0.234375,0.265625,0.265625,0.93359375,0.328125,0.2265625,0.58984375,0.54296875,0.701171875,0.919921875,0.04296875,0.796875,0.240966796875,0.07763671875]),
        # np.array([0.96875,0.578125,0.3515625,0.6328125,0.7578125,0.7109375,0.8984375,0.533203125,0.0703125,0.697265625,0.451171875,0.626953125,0.935546875,0.294921875,0.5244140625]),
        # np.array([0.25,0.421875,0.171875,0.4921875,0.71875,0.51953125,0.71875,0.876953125,0.896484375,0.626953125,0.646484375,0.490234375,0.65234375,0.599609375,0.0341796875]),
        np.array([0.015625,0.015625,0.078125,0.4375,0.59375,0.6953125,0.73828125,0.611328125,0.787109375,0.76171875,0.25,0.427734375,0.154296875,0.592529296875,0.298583984375])
    ]
    # population_evaluation()
    for p in population:
        population_evaluation(population=p)
    end = time.time()
    print(end-start, 's')
    
    
    
    