import argparse
import os, sys
import numpy as np
import glob
import Levenshtein
import time
import torch
from Burgers_1D.test import burgers_1d
from Burgers_2D.test import burgers2d
from Poisson_Classic.test import poisson_classic
from Poisson3D_Complex_Geometry.test import poisson3d_complex_geometry
from Poisson_Equation_With_Subdomains.test import Poisson2DManyArea
from Poisson2D.test import poisson2d
from Poisson3D.test import poisson3d
from Poisson_Boltzmann2D.test import poisson_boltzmann2d
from Heat_Varying_Source.test import heat_varying_source
from Heat_2D_Multi_Scale.test import heat_2d_multi_scale
from Heat_2D_Complex_Geometry.test import heat_complex
from Heat_2D_Long_Time.test import heat_longtime
from NS_Lid_Driven_Flow.test import ns_lid_driven_flow
from NS_Back_Step_Flow.test import ns_2d_back_step_flow
from NS_Long_Time.test import ns_longtime
from Wave_Equation_1D.test import wave
from Wave_Heterogeneous.test import wave_heterogeneous
from Wave_Longtime.test import wave_longtime
from Gray_Scott.test import gray_scott
from Kuramoto_Sivashinsky_Equation.test import kuramoto
from Poisson2D_Inverse.test import poisson_inverse
from Heat2D_Inverse.test import heat2d_inverse
# from Heat_Varing_Source.test import heat_darcy
os.chdir(sys.path[0])

def get_latest_file(folder_path):
    # 获取文件夹中所有文件的列表
    files = glob.glob(os.path.join(folder_path, "*"))

    # 按修改时间从新到旧排序
    files.sort(key=os.path.getmtime, reverse=True)

    # 返回最新文件的地址（如果文件夹非空）
    return files[0] if files else None

def read_single_png_file(folder_path):
    # 获取文件夹中的第一个以.png结尾的文件
    png_file = glob.glob(os.path.join(folder_path, "*.png"))[0]

    # 以二进制方式读取文件
    with open(png_file, "rb") as f:
        content = f.read()

    return os.path.basename(png_file), content

def save_single_png_file(file, output_folder):
    file_name, content = file
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, file_name)
    with open(output_path, "wb") as f:
        f.write(content)
        
# directorys = ['Burgers_1D', 'Burgers_2D', 
#               'Poisson_Classic', 'Poisson_Boltzmann2D', 'Poisson3D_Complex_Geometry',
#               'Heat_2D_Multi_Scale', 
#               'NS_Lid_Driven_Flow', 'NS_Back_Step_Flow', 'NS_Long_Time',
#               'Wave_Equation_1D', 'Wave_Heterogeneous', 
#               'Gray_Scott', 'Kuramoto_Sivashinsky_Equation',
#               'Poisson2D_Inverse', 'Heat2D_Inverse']

# functions = [burgers_1d, burgers2d,
#              poisson_classic, poisson_boltzmann2d, poisson3d_complex_geometry,
#              heat_2d_multi_scale, 
#              ns_lid_driven_flow, ns_2d_back_step_flow, ns_longtime,
#              wave, wave_heterogeneous,
#              gray_scott, kuramoto,
#              poisson_inverse, heat2d_inverse]

directorys = [ 
              'Poisson_Equation_With_Subdomains',
              'Heat_Varying_Source', 'Heat_2D_Complex_Geometry', 'Heat_2D_Long_Time',
              'NS_Long_Time',
              'Wave_Longtime']

functions = [
             Poisson2DManyArea,
             heat_varying_source, heat_complex, heat_longtime,
             ns_longtime,
             wave_longtime]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This is a program to run specific model of different equations.')

    namelist = "', '".join(directorys)
    namelist = "'" + namelist + "'"
    parser.add_argument('-name', type=str, help='The input is expected to a subset of ' + namelist)
    args = parser.parse_args()
    if args.name:
        names = args.name.split(',')
        names = [item.strip() for item in names]
        # print(names)
        indices = []
        RESET = "\033[0m"
        RED = "\033[31m"
        GREEN = "\033[32m"
        for i in range(len(names)):
            try:
                indices.append(directorys.index(names[i]))
            except ValueError as e:
                print('Your input contain model name which cannot be recognized.')
                print(f'Your input:    {RED} {names[i]} {RESET}')
                print(f'Expected input:{GREEN} {min(directorys, key=lambda word: Levenshtein.distance(word, names[i]))} {RESET}')
                exit(0)
    else:
        indices = range(0, len(directorys))
    
with open('result.txt', 'w') as file:
    for i in indices:
        os.chdir('./'+directorys[i])
        mses = []
        l1res = []
        l2res = []
        for j in range(3):
            start_time = time.time()  # 开始时间

            print(f'{directorys[i]}_{j+1}')
            mse, l1re, l2re = functions[i](20000)
            mses.append(mse.reshape(-1, 1))
            l1res.append(l1re.reshape(-1, 1))
            l2res.append(l2re.reshape(-1, 1))

            end_time = time.time()  # 结束时间
            elapsed_time = end_time - start_time  # 计算循环时间
            
            # 将结果写入文件
            file.write(f'Run {directorys[i]}, Iteration {j+1}, MSE: {mse:.5f}, L1RE: {l1re:.5f}, L2RE: {l2re:.5f}, Time: {elapsed_time} seconds\n')

        file.write(f'Run {directorys[i]}, MSE_std: {torch.std(torch.cat(mses)):.5f}, L1RE_std: {torch.std(torch.cat(l1res)):.5f}, L2RE_std: {torch.std(torch.cat(l2res)):.5f}\n')
        os.chdir('./..')

    # for i in indices:
    #     err = [0, 0, 0]
    #     models = [None, None, None]
    #     figs = [None, None, None]
    #     os.chdir('./'+directorys[i])
    #     for j in range(3):
    #         print(f'{directorys[i]}_{j+1}')
    #         print(os.getcwd())
    #         mse, l1re, l2re = functions[i](100)
    #         err[j] = ret.item()
    #         with open(get_latest_file('./model'), "rb") as source_file:
    #             models[j] = source_file.read()
    #         figs[j] = read_single_png_file('.')
    #     err = np.array(err)
    #     min_index = np.argmin(err)
    #     with open(get_latest_file('./model'), "wb") as destination_file:
    #         destination_file.write(models[min_index])
    #     save_single_png_file(figs[min_index], '.')
    #     os.chdir('./..')
    
