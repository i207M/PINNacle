import torch

# def jacobian(u, x):
#     jacobian_matrix = torch.empty((u.shape[1], x.shape[1], x.shape[0]), device=x.device)
#     for u_component in range(u.shape[1]):
#         for x_component in range(x.shape[1]):
#             grad = torch.autograd.grad(u[:, u_component], x, 
#                                        grad_outputs=torch.ones_like(u[:, u_component]),
#                                        create_graph=True,
#                                        only_inputs=True)[0][:, x_component]
#             jacobian_matrix[u_component, x_component] = grad
#     return jacobian_matrix

# def grad(u, x, u_component: int=0, x_component:int=0, order:int=1):
#     jac = jacobian(u, x)
#     selected_grad = jac[u_component, x_component]
#     if order == 1:
#         return selected_grad.unsqueeze(-1)
#     else:
#         return grad(selected_grad.unsqueeze(-1), x, x_component=x_component, order=order - 1)

def grad(u, x, u_component: int=0, x_component:int=0, order:int=1):
    grads = torch.autograd.grad(u[:, u_component], x, 
                                grad_outputs=torch.ones_like(u[:, u_component]),
                                create_graph=True,
                                only_inputs=True)[0]
    grad_specific = grads[:, x_component]
    if order == 1:
        return grad_specific.unsqueeze(-1)
    else:
        return grad(grad_specific.unsqueeze(-1), x, u_component=0, x_component=x_component, order=order - 1)

def jacobian(u, x, i:int=0, j:int=0):
    '''in jacobian, i specifies the u component while j specifies the x component'''
    return grad(u, x, u_component=i, x_component=j)

def hessian(u, x, i:int=0, j:int=0, component: int=0):
    '''in hessian, i specifies the 1st order derivative , j specifies the 2nd order derivative,
       component specifies the dimension of u'''
    return grad(grad(u, x, component, x_component=i), x, 0, x_component=j)