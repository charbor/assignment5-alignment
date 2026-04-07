import torch
from jaxtyping import Float

def compute_entropy(logits: Float[torch.Tensor, "batch seq vocab"]) -> Float[torch.Tensor, "batch seq"]:
    ''' 
    p * log p 
    = p * log( softmax ( logits))
    = p * log(exp(logit) / sum exp logit )
    = p * [log( exp (logit)) - logsumexp (logit)]
    = p * (logit - logsumexp(logit)
    '''
    probs = torch.softmax(logits, dim=-1)
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return - torch.sum(probs * log_probs, dim=-1)

def masked_normalize(
tensor: torch.Tensor,
mask: torch.Tensor,
normalize_constant: float,
dim: int | None= None,
) -> torch.Tensor:
    '''Sum over a dimension and normalize by a constant, considering only those elements where mask == 1.'''
    return torch.sum(tensor * mask, dim=dim) / normalize_constant