# src/dataset.py

import os
import cv2
import torch
from torch.utils.data import Dataset

class ThermalDataset(Dataset):
    def __init__(self, 
                 images_dir, 
                 labels_txt, 
                 transforms=None, 
                 bit_depth=8):
        """
        images_dir:  Path to folder with images (PNG/JPG for 8-bit or TIFF for 16-bit).
        labels_txt:  Path to txt file (e.g. train_labels_8_bit.txt).
        transforms:  Optional transform pipeline (e.g., random flip, etc.).
        bit_depth:   8 or 16 (determine how to read images).
        """
        self.images_dir = images_dir
        self.labels_txt = labels_txt
        self.transforms = transforms
        self.bit_depth = bit_depth
        
        # Read all lines from the label file
        with open(self.labels_txt, 'r') as f:
            lines = f.readlines()
        
        # Each line has format: image_file class_id x_min x_max y_min y_max
        self.annotations = []
        for line in lines:
            parts = line.strip().split()
            # Make sure we parse in correct order:
            image_file = parts[0]
            class_id   = int(parts[1])
            x_min      = float(parts[2])
            x_max      = float(parts[3])
            y_min      = float(parts[4])
            y_max      = float(parts[5])
            
            self.annotations.append({
                "image_file": image_file,
                "class_id": class_id,
                "bbox": [x_min, y_min, x_max, y_max]
            })
        
        # Group annotations by image_file
        self.image_files = sorted(list(set([ann["image_file"] for ann in self.annotations])))

        self.image_to_anns = {}
        for ann in self.annotations:
            f = ann["image_file"]
            if f not in self.image_to_anns:
                self.image_to_anns[f] = []
            self.image_to_anns[f].append(ann)

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Get image filename
        image_file = self.image_files[idx]
        img_path   = os.path.join(self.images_dir, image_file)
        
        # Load the image
        if self.bit_depth == 16:
            # For 16-bit images
            img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
            img = img.astype('float32')
            # Optionally scale to 0..1 range if desired:
            img /= 65535.0
            # shape is (H, W). For PyTorch we want (C, H, W), so let's add channel dim
            img = torch.from_numpy(img).unsqueeze(0)  # shape = (1, H, W)
        else:
            # For 8-bit images
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype('float32') / 255.0
            # Convert from (H, W, C) -> (C, H, W)
            img = torch.from_numpy(img.transpose(2, 0, 1))  # shape = (3, H, W)

        # Gather bounding boxes, labels
        anns = self.image_to_anns[image_file]
        boxes  = []
        labels = []
        for a in anns:
            boxes.append(a["bbox"])       # [x_min, y_min, x_max, y_max]
            labels.append(a["class_id"])  # 1, 2, or 3

        # Convert to tensors
        boxes  = torch.tensor(boxes,  dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"]  = boxes
        target["labels"] = labels
        
        # If we have any transforms (like random flip), apply them
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, image_file
