# src/train.py

import os
import torch
import argparse

from torch.utils.data import DataLoader
import torchvision

from dataset import ThermalDataset
from model import create_model
from transforms import get_transforms

def collate_fn(batch):
    """
    Collate function to handle variable number of objects per image.
    """
    images = []
    targets = []
    files   = []
    for b in batch:
        images.append(b[0])
        targets.append(b[1])
        files.append(b[2])
    return images, targets, files


def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0.0
    
    for images, targets, _ in data_loader:
        # Move everything to GPU/CPU
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass (returns dict of losses)
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backprop
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()

    return total_loss / len(data_loader)

def main(args):
    # Decide on device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Build transforms
    train_tfms = get_transforms(train=True)   # augmentations for train
    val_tfms   = get_transforms(train=False)  # minimal/no augmentation for val

    # Create the dataset
    train_dataset = ThermalDataset(
        images_dir=args.train_images_dir,
        labels_txt=args.train_labels_txt,
        transforms=train_tfms,
        bit_depth=args.bit_depth
    )
    val_dataset = ThermalDataset(
        images_dir=args.val_images_dir,
        labels_txt=args.val_labels_txt,
        transforms=val_tfms,
        bit_depth=args.bit_depth
    )

    # Create the dataloaders
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=2,
                            collate_fn=collate_fn)

    # Create model
    num_classes = 4  # 1 background + 3 classes (person=1, bicycle=2, vehicle=3)
    model = create_model(num_classes=num_classes)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)

    # Training loop
    # Maybe put epoch = 
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device)

        # You could also do validation here if you have a val loop
        # e.g. val_loss = evaluate(model, val_loader) # up to you

        print(f"Epoch [{epoch+1}/{args.epochs}] - Train Loss: {train_loss:.4f}")

    # Save the trained model
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "fasterrcnn_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_images_dir", type=str, required=True,
                        help="Path to training images folder")
    parser.add_argument("--train_labels_txt", type=str, required=True,
                        help="Path to training labels .txt file")
    parser.add_argument("--val_images_dir", type=str, required=True,
                        help="Path to validation images folder")
    parser.add_argument("--val_labels_txt", type=str, required=True,
                        help="Path to validation labels .txt file")
    parser.add_argument("--bit_depth", type=int, default=8,
                        help="8 or 16. Determines how we read images.")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="Directory to save trained model")
    args = parser.parse_args()

    main(args)
