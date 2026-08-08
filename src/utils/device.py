import torch


def resolve_device(requested_device):
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(
            f"warning: requested device '{requested_device}', but CUDA is not "
            "available in this PyTorch build; using CPU instead."
        )
        return torch.device("cpu")
    return device
