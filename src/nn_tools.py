import torch
from torch import Tensor
from jaxtyping import Float, Int


def softmax(x: torch.Tensor, dim: int = -1, tau: float = 1.0) -> torch.Tensor:
    max_el = x.max(dim=dim, keepdim=True)[0]

    return (torch.exp((x - max_el) / tau)) / torch.exp((x - max_el) / tau).sum(dim=dim, keepdim=True)


def clip_gradients(parameters, max_norm: float, eps=1e-6) -> None:
    with torch.no_grad():
        grads = [p.grad.data for p in parameters if p.grad is not None]

        total_norm = torch.sqrt(torch.cat([g.detach().cpu().data.flatten() for g in grads]).pow(2).sum())

        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + eps)
            for g in grads:
                g.detach().mul_(clip_coef)


def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    max_el = inputs.max(dim=-1, keepdim=True)[0]
    stable_logits = inputs - max_el
    selected_stable_logits = torch.gather(stable_logits, dim=-1, index=targets.unsqueeze(-1)).squeeze()

    return (-selected_stable_logits + torch.log(torch.exp(stable_logits).sum(dim=-1))).mean()
