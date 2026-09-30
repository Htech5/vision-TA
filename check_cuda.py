"""Sanity check: confirm torch/ultralytics see CUDA GPU. Run once before training."""
import torch
from ultralytics.utils.torch_utils import select_device

print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"vram: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
select_device("0")
