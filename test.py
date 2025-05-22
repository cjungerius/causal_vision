import torch

def generate_correlated_binary_pairs(batch_size: int, p: float = 0.5, rho: float = 0.0):
    """
    Generate binary (x, y) pairs with:
    - x ~ Bernoulli(p)
    - y has same marginal p
    - Pearson correlation between x and y = rho

    Args:
        batch_size (int): Number of samples
        p (float): P(x=1) and P(y=1)
        rho (float): Desired Pearson correlation between x and y (-1 to 1)

    Returns:
        Tensor of shape (batch_size, 2) with values in {1, 2}
    """
    assert 0 <= p <= 1, "p must be between 0 and 1"
    max_rho = min(p, 1-p) / (p * (1-p))
    assert -max_rho <= rho <= max_rho, f"rho must be in [-{max_rho:.2f}, {max_rho:.2f}] for p={p}"

    # Joint probabilities
    cov = rho * p * (1 - p)
    P11 = p**2 + cov
    P00 = (1 - p)**2 + cov
    P10 = p * (1 - p) - cov
    P01 = p * (1 - p) - cov

    joint_probs = torch.tensor([P00, P01, P10, P11])
    joint_probs = torch.clamp(joint_probs, 0, 1)  # Avoid numerical issues

    dist = torch.distributions.Categorical(joint_probs)
    samples = dist.sample((batch_size,))

    # Map to binary (x, y)
    pair_map = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    pairs = pair_map[samples]

    return pairs + 1  # Optional: shift {0,1} to {1,2}

pairs = generate_correlated_binary_pairs(batch_size=10_000, p=.99, rho=.5)
x, y = pairs[:, 0] - 1, pairs[:, 1] - 1
print(torch.sum(x))
print(torch.sum(y))
print(torch.corrcoef(torch.stack((x.float(), y.float()))))
