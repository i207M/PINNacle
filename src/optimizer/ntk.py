import deepxde as dde
import torch


class LR_Adaptor_NTK(torch.optim.Optimizer):
    """
    Callback for learning rate annealing algorithm of physics-informed neural networks.
    """

    def __init__(self, optimizer, loss_weight, pde):
        '''
        loss_weight - initial loss weights
        num_pde - the number of the PDEs (boundary conditions excluded)
        '''

        super().__init__(optimizer.param_groups, defaults={})
        self.optimizer = optimizer
        self.loss_weight = loss_weight
        self.pde = pde
        self.iter = 0

    @torch.no_grad()
    def step(self, closure):
        self.iter += 1
        with torch.enable_grad():
            inputs = self.pde.data.train_x
            if isinstance(inputs, tuple):
                inputs = tuple(map(lambda x: torch.as_tensor(x).requires_grad_(), inputs))
            else:
                inputs = torch.as_tensor(inputs)
                inputs.requires_grad_()
            outputs = self.pde.net(inputs)
            losses = self.pde.data.losses(targets=None, outputs=outputs, loss_fn=(lambda x, y: (y - x).sum()), inputs=inputs, model=self.pde.model)
            losses = torch.stack(losses)
            loss_r = torch.sum(losses[:self.pde.num_pde])
            loss_b = torch.sum(losses[self.pde.num_pde:])

        m_grad_r = []
        self.zero_grad()
        loss_r.backward(retain_graph=True)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    m_grad_r.append(torch.zeros(p.size))
                else:
                    m_grad_r.append(torch.abs(p.grad).reshape(-1))
        m_grad_r = torch.sum(torch.cat(m_grad_r)**2).item()

        m_grad_b = []
        self.zero_grad()
        loss_b.backward(retain_graph=True)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    m_grad_b.append(torch.zeros(p.size))
                else:
                    m_grad_b.append(torch.abs(p.grad).reshape(-1))
        m_grad_b = torch.sum(torch.cat(m_grad_b)**2).item()

        for i in range(self.pde.num_pde):
            self.loss_weight[i] = (m_grad_r + m_grad_b) / m_grad_r
        for i in range(self.pde.num_pde, len(self.loss_weight)):
            self.loss_weight[i] = (m_grad_r + m_grad_b) / m_grad_b

        return self.optimizer.step(closure)
