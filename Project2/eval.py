import torch
import os
import json
from typing import Tuple, Dict, Any
from torch.utils.data import DataLoader
from torchvision import transforms as T
import numpy as np

# Import your model class (adjust import path if needed)
try:
    from Project2.dual_stream import Network
except Exception:
    from dual_stream import Network


def _read_args_json_if_exists(checkpoint_path: str) -> Dict[str, Any]:
    """Try to read args.json from same directory as checkpoint_path."""
    run_dir = os.path.dirname(checkpoint_path)
    args_path = os.path.join(run_dir, "args.json")
    if os.path.isfile(args_path):
        try:
            with open(args_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def load_model_from_checkpoint(
    ckpt_path: str,
    network_cls,
    device: torch.device = None,
    override_model_kwargs: Dict[str, Any] = None
) -> Tuple[torch.nn.Module, dict]:
    """
    Load model weights from checkpoint and return (model, checkpoint_dict).

    - ckpt_path: path to best_spatial.pth or best_temporal.pth (or best.pth)
    - network_cls: the Network class (callable)
    - device: torch.device to move the model to; defaults to cuda if available else cpu
    - override_model_kwargs: dict to override or provide model constructor kwargs
      e.g. {'num_classes': 101, 'in_channels': 3, 'img_size': 64, 'dropout_rate': 0.0}
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # load checkpoint (map to CPU first to avoid CUDA-device issues)
    ck = torch.load(ckpt_path, map_location="cpu")

    # try to infer model args from args.json in same folder
    inferred = _read_args_json_if_exists(ckpt_path)

    # build model kwargs with fallbacks
    overrides = override_model_kwargs or {}
    model_kwargs = {
        # pick keys from args.json if present, else overrides, else sensible defaults
        "num_classes": overrides.get("num_classes", inferred.get("num_classes", 10)),
        # our constructor uses dropout_rate name in Network; args.json might use 'dropout'
        "dropout_rate": overrides.get("dropout_rate", inferred.get("dropout", 0.0)),
        "img_size": overrides.get("img_size", inferred.get("img_size", 64)),
        "in_channels": overrides.get("in_channels", inferred.get("in_channels", 3)),
    }

    # Instantiate model and load weights
    model = network_cls(
        num_classes=int(model_kwargs["num_classes"]),
        dropout_rate=float(model_kwargs["dropout_rate"]),
        img_size=int(model_kwargs["img_size"]),
        in_channels=int(model_kwargs["in_channels"]),
    )

    # load state dict (allow missing/extra keys tolerance if needed)
    state_dict = ck.get("model_state", ck)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()  # put in eval mode by default

    return model, ck


def evaluate_temporal_on_videos(ckpt_path: str,
                                root_dir: str,
                                eval_split: str = 'val',
                                batch_size: int = 8,
                                workers: int = 4,
                                img_size: int = 64,
                                n_sampled_frames: int = 10,
                                in_channels: int = 18,
                                device: torch.device = None):
    """Load temporal model and run it on FrameVideoDataset validation split.

    Returns: list of softmax probabilities (N, num_classes) and list of labels
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    model, ck = load_model_from_checkpoint(ckpt_path, Network, device=device,
                                           override_model_kwargs={'in_channels': in_channels})

    # transforms for frames (used by dataset to infer target size for flow)
    transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])

    # import FrameVideoDataset
    try:
        from Project2.datasets import FrameVideoDataset
    except Exception:
        from datasets import FrameVideoDataset

    dataset = FrameVideoDataset(root_dir=root_dir, split=eval_split, transform=transform, stack_frames=False, n_sampled_frames=n_sampled_frames)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    softmax = torch.nn.Softmax(dim=1)

    all_probs = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (frames, flows, labels) in enumerate(loader):
            # flows: [batch, C_flow, H, W]
            flows = flows.to(device)
            labels = labels.to(device)

            logits = model(flows)
            probs = softmax(logits)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx+1}/{len(loader)} batches")

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_probs, all_labels, ck



def evaluate_both_on_videos(temporal_ckpt: str,
                            spatial_ckpt: str,
                            root_dir: str,
                            eval_split: str = 'val',
                            batch_size: int = 8,
                            workers: int = 4,
                            img_size: int = 64,
                            n_sampled_frames: int = 10,
                            temporal_in_channels: int = 18,
                            spatial_in_channels: int = 3,
                            device: torch.device = None):
    """Evaluate both spatial and temporal models per video sample, fuse by averaging probabilities,
    and compute per-class accuracies.

    Returns: (t_probs, s_probs, fused_probs, labels, per_class_stats), (t_ck, s_ck)
    where per_class_stats is dict: class -> (spatial_acc, temporal_acc, fused_acc)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load models
    t_model, t_ck = load_model_from_checkpoint(temporal_ckpt, Network, device=device,
                                               override_model_kwargs={'in_channels': temporal_in_channels})
    s_model, s_ck = load_model_from_checkpoint(spatial_ckpt, Network, device=device,
                                               override_model_kwargs={'in_channels': spatial_in_channels})

    # dataset with stacked frames so we can process both
    transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    try:
        from Project2.datasets import FrameVideoDataset
    except Exception:
        from datasets import FrameVideoDataset

    dataset = FrameVideoDataset(root_dir=root_dir, split=eval_split, transform=transform, stack_frames=True, n_sampled_frames=n_sampled_frames)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    softmax = torch.nn.Softmax(dim=1)

    t_all = []
    s_all = []
    fused_all = []
    labels_all = []

    t_model.eval(); s_model.eval()
    with torch.no_grad():
        for batch_idx, (frames, flows, labels) in enumerate(loader):
            # frames: [batch, C, F, H, W], flows: [batch, C_flow, H, W]
            frames = frames.to(device)
            flows = flows.to(device)
            labels = labels.to(device)

            b, c, f, h, w = frames.shape

            # spatial per-frame: run the spatial model on each frame separately,
            # softmax each frame's logits, then average probabilities across frames
            # to obtain a per-video spatial prediction.
            s_probs_accum = None
            for fi in range(f):
                # extract frame fi -> shape (b, c, h, w)
                frame_i = frames[:, :, fi, :, :]
                logits_i = s_model(frame_i)  # (b, num_classes)
                probs_i = softmax(logits_i)  # (b, num_classes)
                if s_probs_accum is None:
                    s_probs_accum = probs_i
                else:
                    s_probs_accum = s_probs_accum + probs_i

            # average over frames
            s_probs = (s_probs_accum / float(f))

            # temporal
            t_logits = t_model(flows)
            t_probs = softmax(t_logits)

            fused = (s_probs + t_probs) / 2.0

            t_all.append(t_probs.cpu().numpy())
            s_all.append(s_probs.cpu().numpy())
            fused_all.append(fused.cpu().numpy())
            labels_all.append(labels.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx+1}/{len(loader)} batches")

    t_all = np.concatenate(t_all, axis=0)
    s_all = np.concatenate(s_all, axis=0)
    fused_all = np.concatenate(fused_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)

    # predictions
    t_preds = np.argmax(t_all, axis=1)
    s_preds = np.argmax(s_all, axis=1)
    fused_preds = np.argmax(fused_all, axis=1)

    # overall accuracies
    t_acc = (t_preds == labels_all).mean() * 100.0
    s_acc = (s_preds == labels_all).mean() * 100.0
    fused_acc = (fused_preds == labels_all).mean() * 100.0
    print(f"Temporal overall acc: {t_acc:.2f}% | Spatial overall acc: {s_acc:.2f}% | Fused overall acc: {fused_acc:.2f}%")

    # per-class accuracies
    classes = np.unique(labels_all)
    per_class_stats = {}
    for cls in classes:
        idx = (labels_all == cls)
        if idx.sum() == 0:
            per_class_stats[int(cls)] = (0.0, 0.0, 0.0)
            continue
        sp_acc = (s_preds[idx] == labels_all[idx]).mean() * 100.0
        tp_acc = (t_preds[idx] == labels_all[idx]).mean() * 100.0
        fu_acc = (fused_preds[idx] == labels_all[idx]).mean() * 100.0
        per_class_stats[int(cls)] = (float(sp_acc), float(tp_acc), float(fu_acc))

    return (t_all, s_all, fused_all, labels_all, per_class_stats), (t_ck, s_ck)


if __name__ == '__main__':
    parse_args_and_run()

