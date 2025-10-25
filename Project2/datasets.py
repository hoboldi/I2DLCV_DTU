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
    transform=None
):
        self.frame_paths = sorted(glob(f'{root_dir}/frames/{split}/*/*/*.jpg'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
       
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

        if self.transform:
            frame = self.transform(frame)
        else:
            frame = T.ToTensor()(frame)

        return frame, label


class FrameVideoDataset(torch.utils.data.Dataset):
    #this is one is used at test time, returns frames, flows, label
    def __init__(self, 
    root_dir = '/dtu/datasets1/02516/ucf101_noleakage', 
    split = 'train', 
    transform = None,
    stack_frames = True,
    n_sampled_frames = 10
    ):
        self.video_paths = sorted(glob(f'{root_dir}/videos/{split}/*/*.avi'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames
        
        self.n_sampled_frames = n_sampled_frames

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
        
        # stack frames into a single tensor with shape (C, F, H, W)
        if self.stack_frames:
            frames = torch.stack(frames).permute(1, 0, 2, 3)

        # also load associated flows for this video (one flow .npy per inter-frame gap)
        flows_dir = video_frames_dir.replace(os.path.join('frames', self.split), os.path.join('flows', self.split))
        flow_files = sorted(glob(os.path.join(flows_dir, '*.npy')))
        expected = max(0, self.n_sampled_frames - 1)
        if len(flow_files) != expected:
            raise FileNotFoundError(f'Video {video_name}: expected {expected} flow files in {flows_dir}, found {len(flow_files)}')

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
                if a.shape[2] in (1,2,3,4) and a.shape[0] > a.shape[2]:
                    a = np.transpose(a, (2,0,1))
                elif a.shape[0] not in (1,2,3,4) and a.shape[2] in (1,2,3,4):
                    a = np.transpose(a, (2,0,1))

            arrays.append(a)

        flow = np.concatenate(arrays, axis=0)  # (C_flow * F, H, W)
        flow_t = torch.from_numpy(flow).float()

        # rescale flow spatially to match transformed frame size if needed
        try:
            # original size from first PIL frame
            orig_w, orig_h = video_frames[0].size
            # transformed frame size from stacked tensor or single frame tensor
            if self.stack_frames:
                new_h, new_w = frames.shape[2], frames.shape[3]
            else:
                new_h, new_w = frames[0].shape[1], frames[0].shape[2]

            f_h, f_w = flow_t.shape[1], flow_t.shape[2]
            if (f_h, f_w) != (new_h, new_w) and (orig_w is not None and orig_h is not None):
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
            pass

        return frames, flow_t, label
    
    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)

        return frames

class FlowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir='/dtu/datasets1/02516/ucf101_noleakage',
        split='train',
        transform=None,
        n_sampled_frames=10
    ):
        self.video_paths = sorted(glob(f'{root_dir}/videos/{split}/*/*.avi'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
        self.n_sampled_frames = n_sampled_frames

    def __len__(self):
        return len(self.video_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = os.path.basename(video_path).replace('.avi', '')
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        # locate flow directory
        flows_dir = video_path.replace('videos', 'flows').replace('.avi', '')
        flow_files = sorted(glob(os.path.join(flows_dir, '*.npy')))

        expected = max(0, self.n_sampled_frames - 1)
        if len(flow_files) != expected:
            raise FileNotFoundError(
                f'Video {video_name}: expected {expected} flow files in {flows_dir}, found {len(flow_files)}'
            )

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
                if a.shape[2] in (1, 2, 3, 4) and a.shape[0] > a.shape[2]:
                    a = np.transpose(a, (2, 0, 1))
                elif a.shape[0] not in (1, 2, 3, 4) and a.shape[2] in (1, 2, 3, 4):
                    a = np.transpose(a, (2, 0, 1))

            arrays.append(a)

        # concatenate flows: (C_flow * F, H, W)
        flow = np.concatenate(arrays, axis=0)
        flow_t = torch.from_numpy(flow).float()

        orig_h, orig_w = flow_t.shape[1], flow_t.shape[2]

        # if transform provided, apply and resize flow accordingly
        if self.transform:
            # get new spatial size from transform if possible
            # apply transform to a dummy image to infer new size
            dummy = Image.fromarray(np.zeros((orig_h, orig_w, 3), dtype=np.uint8))
            transformed_dummy = self.transform(dummy)
            if isinstance(transformed_dummy, torch.Tensor):
                new_h, new_w = transformed_dummy.shape[1], transformed_dummy.shape[2]
            else:
                new_w, new_h = transformed_dummy.size

            if (new_h, new_w) != (orig_h, orig_w):
                flow_t = F.interpolate(
                    flow_t.unsqueeze(0),
                    size=(new_h, new_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

                scale_w = float(new_w) / float(orig_w)
                scale_h = float(new_h) / float(orig_h)
                if abs(scale_w - scale_h) < 1e-6:
                    flow_t = flow_t * scale_w
                else:
                    C = flow_t.shape[0]
                    for c in range(0, C, 2):
                        flow_t[c] *= scale_w
                        if (c + 1) < C:
                            flow_t[c + 1] *= scale_h

        return flow_t, label


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
            
