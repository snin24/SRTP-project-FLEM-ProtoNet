import torch


def euclidean_dist(x, y):
    if x.size(1) != y.size(1):
        raise ValueError("Feature dimensions must match.")

    n = x.size(0)
    m = y.size(0)
    dim = x.size(1)

    if d != y.size(1):
        raise Exception
    
    x = x.unsqueeze(1).expand(n, m, dim)
    y = y.unsqueeze(0).expand(n, m, dim)

    return torch.pow(x - y, 2).sum(dim=2)


def cosine_similarity(x, y):
    if x.size(1) != y.size(1):
        raise ValueError("Feature dimensions must match.")

    x = torch.nn.functional.normalize(x, dim=1)
    y = torch.nn.functional.normalize(y, dim=1)
    return x @ y.transpose(0, 1)
