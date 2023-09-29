import argparse
import time
import os
import re
import Levenshtein as lev
import numpy as np
import torch
import vpinn
from tqdm import tqdm
from difflib import SequenceMatcher
from src.config.default_arg import argContainer
from src.pde.burgers import Burgers1d, Burgers2d
from src.pde.poisson import Poisson2d, Poisson_boltzmann2d, Poisson3d, Poisson2d_Many_subdomains
from src.pde.heat import Heat2d_Varying_Source, Heat_Multi_scale, HeatComplex, HeatLongTime
from src.pde.ns import NSEquation_LidDriven, NS_Back_Step, NSEquation_Long
from src.pde.wave import WaveEquation1D, WaveHeterogeneous, WaveEquation2D_Long
from src.pde.chaotic import GrayScottEquation, KuramotoSivashinskyEquation
from src.pde.inverse import PoissonInv, HeatInv
parser = argparse.ArgumentParser(description='PINNBench trainer')
parser.add_argument('--name', type=str, default="benchmark")
parser.add_argument('--case', type=str, default='all', help="input testcase name")
parser.add_argument('--device', type=str, default="0")
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('--hidden_layers', type=str, default="default")
parser.add_argument('--loss_weight', type=str, default="default")
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--iter', type=int, default=10000)
parser.add_argument('--logevery', type=int, default=100)
parser.add_argument('--plotevery', type=int, default=2000)
parser.add_argument('--error_scale', type=str, default='mse, l1re, l2re')
parser.add_argument('--ablation', type=bool, default=False) # ablation argument is not available for ordinary users
parser.add_argument('--params', type=bool, default=False) # params argument is not available for ordinary users

command_args = parser.parse_args()

date_str = time.strftime('%m.%d-%H.%M.%S', time.localtime())

def parse_hidden_layers():
    layers = []
    for s in re.split(r"[, _-]",command_args.hidden_layers):
        if '*' in s:
            siz, num = s.split('*')
            layers += [int(siz)] * int(num)
        else:
            layers += [int(s)]
    return layers

def parse_loss_weight():
    if command_args.loss_weight == '': return None
    weights = []
    for s in re.split(r"[, _-]",command_args.loss_weight):
        weights.append(float(s))
    return weights

def extract_numbers(s):
    # use regular expressions to find all the numbers in the string and convert them to a list of integers
    return [int(n) for n in re.findall(r'\d+', s)]

pde_list = \
    [Burgers1d, Burgers2d] + \
    [Poisson2d, Poisson_boltzmann2d, Poisson3d, Poisson2d_Many_subdomains] + \
    [Heat2d_Varying_Source, Heat_Multi_scale, HeatComplex, HeatLongTime] + \
    [NSEquation_LidDriven, NS_Back_Step, NSEquation_Long] + \
    [WaveEquation1D, WaveHeterogeneous, WaveEquation2D_Long] + \
    [GrayScottEquation, KuramotoSivashinskyEquation] + \
    [PoissonInv, HeatInv]

pde_name_list = [case.__name__ for case in pde_list]

def find_most_similar(input_str, list_strings):
    if input_str == 'all':
        return list(range(len(pde_list)))
    
    # use regular expressions to split the input string with characters other than letters and numbers as delimiters
    substrings = re.split(r'\W+', input_str)
    # store an index list of results
    result_indices = []

    # iterate over each substring and find the index of the most similar string in the list
    for substring in substrings:
        distances = [lev.distance(substring, list_string) for list_string in list_strings]
        result_indices.append(distances.index(min(distances)))

    return result_indices

task_list = find_most_similar(command_args.case, pde_name_list)

for i in task_list:
    if command_args.hidden_layers != 'default':
        getattr(argContainer, pde_name_list[i]).set_layer_size([pde_list[i]().indim] + parse_hidden_layers() + [pde_list[i]().outdim])
    if command_args.loss_weight != 'default':
        getattr(argContainer, pde_name_list[i]).set_coef(parse_loss_weight())
    getattr(argContainer, pde_name_list[i]).set_epoch_num(command_args.iter)
    
            
argContainer.to_json()

device_list = extract_numbers(command_args.device)

def set_seed(seed:int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if command_args.seed != -1:
    print(f'set seed')
    set_seed(command_args.seed)

def is_similar(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio() > 0.8

def find_matches(input_str, match_list):
    # use regular expressions to split the input string, with non-alphanumeric characters used as delimiters
    input_words = re.split(r'\W+', input_str.lower())
    
    matches = []
    for word in input_words:
        for index, match in enumerate(match_list):
            if is_similar(word, match):
                matches.append(index)
                break
            
    return matches

error_scale = [vpinn.loss.mse_loss, vpinn.loss.l1re_loss, vpinn.loss.l2re_loss]
error_scale_name = ['mse', 'l1re', 'l2re']
pdebench_error_name = ['err_RMSE', 'err_nRMSE', 'err_CSV', 'err_Max', 'err_F']

selected_error_scale = find_matches(command_args.error_scale, error_scale_name)

from multiprocessing import Manager
from threading import Thread
import gc
import ablation
import params

if command_args.ablation:
    task_list = ablation.gen_task_list()
    arg_List = ablation.gen_arg_list()

if command_args.params:
    task_list = params.gen_task_list()
    params_list = params.params_list()

def run_and_plot(device, i, queue, lr, logevery, plotevery, gpu_status, args=None):
    gpu_status[device] = 'busy'
    if command_args.params:
        thread = pde_list[task_list[i]](**params_list[i], device=device)
    else:
        thread = pde_list[task_list[i]](device=device)
    thread.run(None, i, queue, lr, logevery, plotevery, args)
    result = thread.plot()
    with open(f'./log/task_id:{i}, {pde_list[task_list[i]].__name__}.txt', 'a') as file:
        for j in range(len(result)):
            file.write(f'model {j}:\n')
            for i in range(len(selected_error_scale)):
                # measure the error of all models
                file.write(f'{error_scale_name[selected_error_scale[i]]} error: {error_scale[selected_error_scale[i]](*result[j])}' + '\n')
        
            pdebench_error = vpinn.pdebench_err.metric_func(*result[j])
            for i in range(len(pdebench_error_name)):
                file.write(f'{pdebench_error_name[i]}: {pdebench_error[i]}' + '\n')
            
    gpu_status[device] = 'free'
    del thread
    gc.collect()
    torch.cuda.empty_cache()

def scheduler(task_list, gpu_status, queue):
    for i in range(len(task_list)):
            
        while True:
            for gpu, status in gpu_status.items():
                if status == 'free':
                    device = gpu
                    gpu_status[device] = 'busy'
                    if command_args.ablation:
                        p = torch.multiprocessing.Process(target=run_and_plot, args=(device, i, queue, command_args.lr, command_args.logevery, command_args.plotevery, gpu_status, arg_List[i]))
                    else :
                        p = torch.multiprocessing.Process(target=run_and_plot, args=(device, i, queue, command_args.lr, command_args.logevery, command_args.plotevery, gpu_status))
                    p.daemon = True
                    p.start()
                    processes.append(p)
                    break
            else:
                time.sleep(1)  # Wait for a GPU to be free
                continue
            break
        time.sleep(0.2)
    
    # print('scheduler finished')
    exit(0)
    
manager = Manager()
gpu_status = manager.dict({f'cuda:{device_list[i]}': 'free' for i in range(len(device_list))})

processes = []
queue = torch.multiprocessing.Queue()

torch.cuda.empty_cache()

# Start the scheduler thread
scheduler_thread = Thread(target=scheduler, args=(task_list, gpu_status, queue))
scheduler_thread.start()

pbar_dict = {i: tqdm(total=command_args.iter, position=i, desc=f"{pde_name_list[task_list[i]]}") for i in range(len(task_list))}

for _ in range(command_args.iter * len(task_list)):
    task_id, progress = queue.get()
    pbar_dict[task_id].update(progress - pbar_dict[task_id].n)
    
for p in processes:
    p.join()

scheduler_thread.join()
