import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.95), eps=1e-8):
        if lr < 0:
            raise ValueError(f'Invalid leraning rate: {lr}')
        defaults = {'lr': lr,
                    'weight_decay': weight_decay,
                    'betas': betas,
                    'eps': eps}
        super().__init__(params, defaults)


    def step(self, closure=None):
        loss = None if closure is None else closure()
        with torch.no_grad():
            for group in self.param_groups:
                lr = group['lr']
                weight_decay = group['weight_decay']
                b1, b2 = group['betas']
                eps = group['eps']

                for p in group['params']:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    grad = p.grad.data # Get the gradient of the loss

                    if len(state) == 0:
                        state['t'] = 1
                        state['m'] = torch.zeros_like(p.data)
                        state['v'] = torch.zeros_like(p.data)

                    t = state['t']
                    m = state['m']
                    v = state['v']

                    m.mul_(b1).add_(grad, alpha=1-b1) # Update the first moment estimate
                    v.mul_(b2).addcmul_(grad, grad, value=1-b2) # Update the second moment estimate

                    lr_t = lr * (1 - b2**t)**0.5 / (1 - b1**t) # Compute adjusted lr for iteration t

                    p.data.mul_(1 - lr * weight_decay) # Apply weight decay

                    denom = v.sqrt().add(eps)
                    p.data.addcdiv_(m, denom, value=-lr_t) # Update the parameters

                    state['t'] = t + 1

        return loss


class CosineWithWarmup():
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            max_lr: float,
            min_lr: float,
            T_c: int,
            T_w: int = 0
    ):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.T_w = T_w
        self.T_c = T_c
        self.t = 0
        self.cur_st = 1
        self.warmup_t = 0


    def step(self):
        if self.warmup_t <= self.T_w:
            self.warmup_t += 1
            lr = self.warmup_t * self.max_lr / self.T_w
        else:
            self.t += self.cur_st

            lr = self.min_lr + 0.5 * (1 + math.cos(math.pi * self.t / self.T_c)) * (self.max_lr - self.min_lr)

            if self.t == self.T_c:
                self.cur_st = -1
            elif self.t == 1:
                self.cur_st = 1

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f'Invalid leraning rate: {lr}')
        defaults = {'lr': lr}
        super().__init__(params, defaults)


    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get('t', 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state['t'] = t + 1

        return loss