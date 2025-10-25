import argparse
import time
import random
from pathlib import Path


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from Project2.dual_stream import Network
    from Project2.datasets import FrameImageDataset
    from Project2.datasets import FlowDataset
except Exception:
    # allow running when this file is executed from Project2/ directly
    from dual_stream import Network
    from datasets import FrameImageDataset
    from datasets import FlowDataset

