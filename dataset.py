import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob

class VideoFrameDataset(Dataset):
    def __init__(self, root_dir, sequence_length=5, transform=None):
        """
        Args:
            root_dir (string): Directory with all the video folders.
            sequence_length (int): How many frames to consider in one sample 
                                   (e.g., 4 past frames + 1 target frame).
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.samples = []

        # 1. Index the data
        # We loop through every video folder and calculate possible sequences
        video_folders = sorted(os.listdir(root_dir))
        
        for vid_folder in video_folders:
            vid_path = os.path.join(root_dir, vid_folder)
            if not os.path.isdir(vid_path):
                continue
                
            # Get all frames in this video folder, sorted numerically
            frames = sorted(glob.glob(os.path.join(vid_path, "*.png")))
            
            # Create sliding window sequences
            # If we have N frames and sequence_length L, we can make N - L + 1 sequences
            if len(frames) >= sequence_length:
                for i in range(len(frames) - sequence_length + 1):
                    # Store the path to the frames, not the images themselves (to save RAM)
                    self.samples.append(frames[i : i + sequence_length])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 1. Retrieve the paths for this specific sequence
        frame_paths = self.samples[idx]
        
        # 2. Load images and stack them
        frames = []
        for path in frame_paths:
            img = Image.open(path).convert('RGB') # or 'L' for grayscale
            if self.transform:
                img = self.transform(img)
            frames.append(img)
            
        # 3. Stack list of Tensors into one Tensor
        # Current shape of list: [ (C, H, W), (C, H, W), ... ]
        # Desired shape: (Time, C, H, W)
        frames_tensor = torch.stack(frames)
        
        # 4. Split into Input (Past) and Target (Future)
        # Input: All frames except the last one
        # Target: The last frame (what we want to predict)
        input_seq = frames_tensor[:-1] 
        target_frame = frames_tensor[-1]
        
        return input_seq, target_frame