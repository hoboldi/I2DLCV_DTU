from glob import glob
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T
import numpy as np
import torch.nn.functional as F

class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir='/dtu/datasets1/02516/ucf101_noleakage',
    split='train', 
    transform=None,
    flow_transform =None,
    use_flow=False,
    flow_root=None,
    n_sampled_frames=10,
    ):
        self.root_dir = root_dir
        self.frame_paths = sorted(glob(f'{root_dir}/frames/{split}/*/*/*.jpg'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
        # flow options
        self.use_flow = use_flow
        self.flow_root = flow_root
        self.n_sampled_frames = n_sampled_frames
        self.flow_transform = flow_transform
       
        # precompute mapping video_name -> sorted list of flow .npy files
        self.video_to_flow_files = {}
        expected = max(0, self.n_sampled_frames - 1)
        for frame_path in self.frame_paths:
            video_dir = os.path.dirname(frame_path)
            video_name = os.path.basename(video_dir)
            if video_name in self.video_to_flow_files:
                continue

            # compute flows_dir by mirroring frames/<split> -> flows/<split>
            if self.flow_root:
                rel = os.path.relpath(video_dir, os.path.join(self.root_dir, 'frames', split))
                flows_dir = os.path.join(self.flow_root, split, rel)
            else:
                flows_dir = video_dir.replace(os.path.join('frames', split), os.path.join('flows', split))

            flow_files = sorted(glob(os.path.join(flows_dir, '*.npy')))
            if len(flow_files) != expected:
                raise FileNotFoundError(f'Video {video_name}: expected {expected} flow files in {flows_dir}, found {len(flow_files)}')

            # optional: could validate shapes/dtypes here if desired
            self.video_to_flow_files[video_name] = flow_files

    def __len__(self):
        return len(self.frame_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        video_name = frame_path.split('/')[-2]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()
        
        frame = Image.open(frame_path).convert("RGB")
        orig_w, orig_h = frame.size

        if self.transform:
            frame = self.transform(frame)
        else:
            frame = T.ToTensor()(frame)

        if not self.use_flow:
            return frame, label

        # load associated flows
        video_dir = os.path.dirname(frame_path)
        # map frames/.../class/video -> flows/.../class/video
        flows_dir = video_dir.replace(os.path.join('frames', self.split), os.path.join('flows', self.split))

        # look up precomputed list of flow files for this video
        flow_files = self.video_to_flow_files.get(video_name)
        if flow_files is None:
            raise FileNotFoundError(f'No precomputed flow files for video {video_name} (looked in {flows_dir})')

        arrays = []
        for f in flow_files:
            a = np.load(f)
            if np.issubdtype(a.dtype, np.integer):
                a = a.astype(np.float32) / 255.0
            else:
                a = a.astype(np.float32)

            # ensure channels-first
            if a.ndim == 2:
                a = a[np.newaxis, :, :]
            elif a.ndim == 3:
                # if channels last (H,W,C) -> transpose
                if a.shape[2] in (1,2,3,4) and a.shape[0] > a.shape[2]:
                    a = np.transpose(a, (2,0,1))
                elif a.shape[0] not in (1,2,3,4) and a.shape[2] in (1,2,3,4):
                    a = np.transpose(a, (2,0,1))

            arrays.append(a)

        flow = np.concatenate(arrays, axis=0)  # (C_flow * F, H, W)
        flow_t = torch.from_numpy(flow).float()
        # If the image transform resized the image, rescale flow displacement values
        # so they remain correct in the transformed image coordinate system.
        # Use the transformed frame tensor size (C, H, W) to get new H,W.
        try:
            new_h, new_w = frame.shape[1], frame.shape[2]
            f_h, f_w = flow_t.shape[1], flow_t.shape[2]
            if (f_h, f_w) != (new_h, new_w) and (orig_w is not None and orig_h is not None):
                # spatially resample flow and then scale dx/dy channels by image scale
                flow_t = F.interpolate(flow_t.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
                scale_w = float(new_w) / float(orig_w)
                scale_h = float(new_h) / float(orig_h)
                if abs(scale_w - scale_h) < 1e-6:
                    flow_t = flow_t * scale_w
                else:
                    C = flow_t.shape[0]
                    for c in range(0, C, 2):
                        flow_t[c] = flow_t[c] * scale_w
                        if (c + 1) < C:
                            flow_t[c + 1] = flow_t[c + 1] * scale_h
        except Exception:
            # if anything goes wrong, fall back to using the flow as-is
            pass
        if self.flow_transform:
            flow_t = self.flow_transform(flow_t)

        return frame, flow_t, label


class FrameVideoDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir = '/dtu/datasets1/02516/ucf101_noleakage', 
    split = 'train', 
    transform = None,
    stack_frames = True,
    use_flow = False,
    flow_root = None,
    n_sampled_frames = 10
    ):
        self.video_paths = sorted(glob(f'{root_dir}/videos/{split}/*/*.avi'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames
        
        self.n_sampled_frames = n_sampled_frames
        # flow-related
        self.use_flow = use_flow
        self.flow_root = flow_root

    def __len__(self):
        return len(self.video_paths)
    
    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = video_path.split('/')[-1].split('.avi')[0]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        video_frames_dir = self.video_paths[idx].split('.avi')[0].replace('videos', 'frames')
        video_frames = self.load_frames(video_frames_dir)
        if self.transform:
            frames = [self.transform(frame) for frame in video_frames]
        else:
            frames = [T.ToTensor()(frame) for frame in video_frames]
        
        if self.stack_frames:
            frames = torch.stack(frames).permute(1, 0, 2, 3)

        return frames, label
    
    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)

        return frames



if __name__ == '__main__':
    from torch.utils.data import DataLoader

    root_dir = '/dtu/datasets1/02516/ucf101_noleakage'

    transform = T.Compose([T.Resize((64, 64)),T.ToTensor()])
    frameimage_dataset = FrameImageDataset(root_dir=root_dir, split='val', transform=transform)
    framevideostack_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = True)
    framevideolist_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = False)


    frameimage_loader = DataLoader(frameimage_dataset,  batch_size=8, shuffle=False)
    framevideostack_loader = DataLoader(framevideostack_dataset,  batch_size=8, shuffle=False)
    framevideolist_loader = DataLoader(framevideolist_dataset,  batch_size=8, shuffle=False)

    # for frames, labels in frameimage_loader:
    #     print(frames.shape, labels.shape) # [batch, channels, height, width]

    # for video_frames, labels in framevideolist_loader:
    #     print(45*'-')
    #     for frame in video_frames: # loop through number of frames
    #         print(frame.shape, labels.shape)# [batch, channels, height, width]

    for video_frames, labels in framevideostack_loader:
        print(video_frames.shape, labels.shape) # [batch, channels, number of frames, height, width]
            
