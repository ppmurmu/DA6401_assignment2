from tqdm.auto import tqdm
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import wandb
from torch.utils.data.sampler import SubsetRandomSampler
from pathlib import Path
import torchvision.models as models
from sklearn.model_selection import StratifiedShuffleSplit
import argparse

#------------DATA LOADER-----------------


def create_dataloaders(
    dataset_path: str,
    num_classes: int = 10,
    data_augmentation: bool = False,
    batch_size: int = 64,
    val_ratio: float = 0.2,
    num_workers: int = 2,
    seed: int = 42,
    resize: tuple[int, int] = (224, 224), #googlenet
):
    # 1) Normalization constants
    mean = [0.4712, 0.46, 0.389]
    std  = [0.19, 0.18, 0.18]

    # 2) Build a single train transform (conditional augment) + a test transform
    # ----------------------------------------------------------------------------
    train_ops = [
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    if data_augmentation:
        # insert augment ops *before* ToTensor
        aug_ops = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
        ]
        train_ops[0:0] = aug_ops  # prepend aug ops to the pipeline

    train_transform = transforms.Compose(train_ops)

    test_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # 3) Load the full train & test datasets once
    train_dir = Path(dataset_path) / "train"
    val_dir   = Path(dataset_path) / "val"

    full_train = datasets.ImageFolder(train_dir, transform=train_transform)
    test_ds    = datasets.ImageFolder(val_dir,   transform=test_transform)

    # 4) Stratified split: ensures each class is represented in train & val
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_ratio,
        random_state=seed
    )
    train_idx, val_idx = next(splitter.split(full_train.samples, full_train.targets))

    train_ds = Subset(full_train, train_idx)

    # For val we want the *same* Resize+Normalize but **no** augmentation:
    val_base = datasets.ImageFolder(train_dir, transform=test_transform)
    val_ds   = Subset(val_base, val_idx)

    # 5) DataLoader kwargs for speed
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    # 6) Return loaders + class names
    return train_loader, val_loader, test_loader

#-----------------TRAIN & VAL-----------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, count = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="  ▶ Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        count   += imgs.size(0)

    avg_loss = total_loss / count
    accuracy = correct / count * 100
    return avg_loss, accuracy


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="  ▶ Validating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            count   += imgs.size(0)

    avg_loss = total_loss / count
    accuracy = correct / count * 100
    return avg_loss, accuracy

#-----------TRAINING MODEL---------------
# Function to train the model
def train_model(model, train_loader, val_loader, loss_func, optimizer, device, num_epochs=5):
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_func, device)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')

        val_loss, val_acc =  validate(model, val_loader, loss_func, device)

        print(f'Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
        wandb.log({'train_loss': train_loss, 'train_acc': train_acc, 'val_loss' : val_loss, 'val_acc' : val_acc, 'epochs' : epoch + 1})


#---------------FINE TUNING--------------------
# Fine-tuning function
def fine_tune(model, strategy='freeze_all'):
    if strategy == 'freeze_all':
        # Freeze all layers except the final classification layer
        for name, param in model.named_parameters():
            if "fc" not in name:  # Skip parameters of the final fully connected layer
                param.requires_grad = False
    elif strategy == 'unfreeze_all':
        # Unfreeze all layers and train the entire model
        for param in model.parameters():
            param.requires_grad = True
    elif strategy == 'freeze_first_5':
        # Unfreeze and fine-tune only a subset of layers (e.g., only top layers)
        for i, param in enumerate(model.parameters()):
            if i < 5:
                param.requires_grad = False
    elif strategy == 'freeze_first_15':
        # Unfreeze and fine-tune only a subset of layers (e.g., only top layers)
        for i, param in enumerate(model.parameters()):
            if i < 15:
                param.requires_grad = False

    return model


sweep_config = {
    'method': 'grid',
    'name' : 'part-b-googlenet',
    'metric': {
      'name': 'val_acc',
      'goal': 'maximize'
    },
    'parameters': {
        'strategy': {
          'values': ["freeze_all", "unfreeze_all", "freeze_first_5", "freeze_first_15"]
        },
        'learning_rate': {
            'values':[1e-3,1e-4]
        },
        'epoch' : {
            'values' : [5,7]
        }
    }
}




def main():
    run = wandb.init()

    config = wandb.config

    run_name = f"googlenet_strat={config.strategy}_lr={config.learning_rate}"
    # Set the run name
    run.name = run_name

    # Load pre-trained model
    model = models.googlenet(pretrained=True)

    num_classes = 10  # Number of classes in iNaturalist dataset

    num_features = model.fc.in_features

    model.fc = nn.Linear(num_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = create_dataloaders(dataset_path=directory, data_augmentation=True)

    # Define loss function and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # fine tuning step
    model = fine_tune(model, strategy=config.strategy )
    model.to(device)

    train_model(model, train_loader, val_loader, loss_func, optimizer, device, num_epochs=config.epoch)



#argument parser for script
def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN with customizable hyperparameters")
    parser.add_argument("-dr", "--dir",type=str, default="./inaturalist_12K/")
    parser.add_argument("-wp", "--wandb_project", type=str, default="DA6401-A2", help="Weights & Biases project name")
    parser.add_argument("-we", "--wandb_entity", type=str, default="cs24m033-iit-madras", help="Weights & Biases entity")
    parser.add_argument("-e", "--epoch", type=int, default=1, help="Number of training epochs")   
    parser.add_argument("-st", "--strategy", type=str, default="freeze_all", help="strategy")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")

    return parser.parse_args()


def create_sweep_config(args):
    return {
        "method": "grid",
        "name": "googlenet",
        "metric": {
            "goal": "maximize",
            "name": "val_acc"
        },
        "parameters": {
            "strategy": {"value": args.strategy},
            "learning_rate": {"value": args.learning_rate},
            "epoch" : {"value": args.epoch}
        }
    }

if __name__ == "__main__":
    wandb.login()
    args = parse_args()
    sweep_config = create_sweep_config(args)
    directory = args.dir
    wandb_id = wandb.sweep(sweep_config,entity=parse_args().wandb_entity, project=parse_args().wandb_project)

    #change count value to run experiments
    wandb.agent(wandb_id, function=main, count=1)



