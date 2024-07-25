import pickle
from argparse import ArgumentParser
import json
import copy
import requests
from http.server import HTTPServer, BaseHTTPRequestHandler
from record import Record
import numpy as np




def evaluation(records, args, n_measure):
    ratio_records_dict = {}
    for r in records:
        ratio = tuple(r.ratio)
        ratio_records_dict[ratio] = r
    
    # duplicate n_mearsure times 
    population = ratio_records_dict.keys()
    pop = list(population)
    val_pop = []
    for p in pop:
        for _ in range(n_measure):
            tmp_p = copy.deepcopy(p)
            val_pop.append(p)
    server = "http://localhost:9001/submit"
    fn = ["loss", "mem", "lat", "mmin", "mmax"]
    callback_address = "http://localhost:8080"
    model = args.model_type
    data = {
        "pop": val_pop,
        "fn": fn,
        "args": {
            "callback_address": callback_address,
            "model": model,
        },
    }
    KEEP_RUNING = [True]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            post_data = self.rfile.read(int(self.headers['content-length']))
            msg = json.loads(post_data.decode())
            print("receive metrics from server: ", msg)
            try:
                metrics = msg['metrics']
                for metric in metrics:
                    idx = metric['idx']
                    key = val_pop[idx]
                    
                    device = metric['device']
                    record = ratio_records_dict[key]
                    if device not in record.memory.keys():
                        record.memory[device] = []
                        record.latency[device] = []
                        
                    record.memory[device].append(metric['mem'])
                    record.latency[device].append(metric['lat'])

                self.response_data = {
                    'msg': "result received. good.",
                    "status_code": 200
                }
                self.response_code = 200
                keep_running(False)
            except Exception as e:
                print(f"Exception occurred: {e}")
                self.response_data = {
                    'msg': "result received. wrong format result.",
                    "status_code": 200
                }
                self.response_code = 401
            self.send_response(self.response_code)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(self.response_data).encode('utf-8'))

    def keep_running(flag=None):
        if flag is not None:
            KEEP_RUNING[0] = flag
        return KEEP_RUNING[0]

    req = requests.post(server, data=json.dumps(data))
    status_code = req.status_code
    msg = req.text
    print("resp: ", msg)

    # 异步，修改为监听一个端口号
    if status_code == 200:
        host = ('localhost', 8080)
        server = HTTPServer(host, Handler)
        while keep_running():
            server.handle_request()
    else:
        print("wrong request")

def load_records(file_path):
    with open(file_path, 'rb') as f:
        records = pickle.load(f)    
    return records


def divide_records(records, file_path, args, device_num=1):
    group_num = len(records) // device_num
    
    ckpt = 0
    
    for i in range(group_num):
        if i < ckpt:
            continue
        start = i*device_num
        end = (i+1)*device_num
        grouped_record = records[start:end]
        evaluation(grouped_record, args, 10)
        with open(file_path, "wb") as f:
            pickle.dump(records, f)
        with open('output.txt', 'a') as f:
            f.write(f"[0:{end}) records successful!\n")
    
    with open('output.txt', 'a') as f:
        f.write("All Successful!!\n")
    


if __name__=='__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-m','--model_type', type=str, default='alexnet')
    args = arg_parser.parse_args()
    
    file_path = "vgg16-5000-10.pkl"
    # file_path = 'resnet50-5000-10-test.pkl'
    
    print(file_path)

    records = load_records(file_path)
    
    divide_records(records, file_path, args)
    
