import torch

def del_tensor_ele(tensor,index):
    t1 = tensor[0:index]
    t2 = tensor[index+1:]
    return torch.cat((t1,t2),dim=0)