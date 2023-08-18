import vpinn
from  src.config.default_arg import ArgContainer

# (3 times, 4 trials, 3 arguments: grid_num, Q, test_fcn_num)
len = 4 * 2

def gen_task_list():
    return [16] * len

def gen_arg_list():
    arg_list = []
    for i in range(len):
        arg_list.append(ArgContainer())
    arg_list[0].GrayScottEquation.grid_num = [3] * 3
    arg_list[1].GrayScottEquation.grid_num = [4] * 3
    arg_list[2].GrayScottEquation.grid_num = [5] * 3
    arg_list[3].GrayScottEquation.grid_num = [6] * 3
    arg_list[4].GrayScottEquation.Q = 6
    arg_list[5].GrayScottEquation.Q = 8
    arg_list[6].GrayScottEquation.Q = 10
    arg_list[7].GrayScottEquation.Q = 12
    
    return arg_list