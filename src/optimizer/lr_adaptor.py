import torch


class LR_Adaptor(torch.optim.Optimizer):
    """
    Callback for learning rate annealing algorithm of physics-informed neural networks.
    """

    def __init__(self, optimizer, loss_weight, num_pde, alpha=0.1, mode="max"):
        '''
        loss_weight - initial loss weights
        num_pde - the number of the PDEs (boundary conditions excluded)
        alpha - parameter of moving average
        mode - "max" (PINN-LA), "mean" (PINN-LA-2)
        '''
        assert mode == "max"

        defaults = dict(alpha=alpha, mode=mode)
        super().__init__(optimizer.param_groups, defaults)
        self.optimizer = optimizer
        self.loss_weight = loss_weight
        self.num_pde = num_pde
        self.alpha = alpha
        self.mode = mode
        self.iter = 0

    @torch.no_grad()
    def step(self, closure):
        self.iter += 1
        with torch.enable_grad():
            _ = closure(skip_backward=True)
            losses = self.losses / torch.as_tensor(self.loss_weight)  # get non-weighted loss from closure
            pde_loss = torch.sum(losses[:self.num_pde])

        self.zero_grad()
        pde_loss.backward(retain_graph=True)
        m_grad_r = -torch.inf
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    m_grad_r = max(m_grad_r, torch.max(torch.abs(p.grad)).item())

        # adapt the weights for each bc term
        for i in range(self.num_pde, len(self.loss_weight)):
            sum = 0
            count = 0
            with torch.enable_grad():
                self.zero_grad()
                losses[i].backward(retain_graph=True)
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        sum += torch.sum(torch.abs(p.grad))
                        count += torch.numel(p.grad)
            mean = sum / count
            lambda_hat = m_grad_r / (mean * self.loss_weight[i])
            self.loss_weight[i] = (1 - self.alpha) * self.loss_weight[i] + self.alpha * lambda_hat

        with torch.enable_grad():
            self.zero_grad()
            total_loss = torch.sum(losses * torch.as_tensor(self.loss_weight))
            total_loss.backward()
        self.optimizer.step()

        return total_loss
