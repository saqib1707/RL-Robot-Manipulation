# -*- coding: utf-8 -*-
from torch import optim


# Non-centered RMSprop update with shared statistics (without momentum)
class SharedRMSprop(optim.RMSprop):
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0):
        """
        Optimiser constructor.

        Args:
            params (iterable): iterable of module parameters.
            lr (float, optional): learning rate. Defaults to 1e-2.
            alpha (float, optional): RMSprop decay factor. Defaults to 0.99.
            eps (float, optional): term added to the denominator to improve numerical stability. Defaults to 1e-8.
            weight_decay (float, optional): weight decay (L2 penalty) . Defaults to 0.
        """
        super(SharedRMSprop, self).__init__(
            params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=0, centered=False
        )
        # State initialization
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = p.data.new().resize_(1).zero_()
                state["square_avg"] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        """
        Method to update the share memory.
        """
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"].share_memory_()
                state["square_avg"].share_memory_()

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): reevaluates the model and returns the loss. Defaults to None.

        Returns:
            float: loss value
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                square_avg = state["square_avg"]
                alpha = group["alpha"]
                state["step"] += 1
                if group["weight_decay"] != 0:
                    grad = grad.add(group["weight_decay"], p.data)
                # g = αg + (1 - α)Δθ^2
                # square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)   # deprecated addcmul_ signature
                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                # θ ← θ - ηΔθ/√(g + ε)
                avg = square_avg.sqrt().add_(group["eps"])
                # p.data.addcdiv_(-group["lr"], grad, avg)   # deprecated addcdiv_ signature
                p.data.addcdiv_(grad, avg, value=-group["lr"])
        return loss
