import argparse
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
from sklearn.model_selection import StratifiedShuffleSplit

#------------------DATA LOADERS--------------------
def create_dataloaders(
    dataset_path: str = "inaturalist_12K/",
    num_classes: int = 10,
    data_augmentation: bool = False,
    batch_size: int = 64,
    val_ratio: float = 0.2,
    num_workers: int = 2,
    seed: int = 42,
    resize: tuple[int, int] = (256, 256),
):
    # 1) Mean and standard deviation values calculated from function get_mean_and_std on training dataset
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
        # insert augment ops before ToTensor
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

    # For val we want the same Resize+Normalize but no augmentation:
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
    return train_loader, val_loader, test_loader, full_train.classes

#--------------------------MODEL-----------------------

class SimpleCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_filters: list[int],
        filter_sizes: list[int],
        activation_fn: nn.Module,
        hid_neurons: int,
        dropout: float = 0.0,
        use_batch_norm: bool = False
    ):
        super(SimpleCNN, self).__init__()
        layers = []
        in_channels = 3 # RGB input

        # Build convolutional feature extractor
        for out_ch, k in zip(num_filters, filter_sizes):
            layers.append(nn.Conv2d(in_channels, out_ch, kernel_size=k, padding=k//2))
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(activation_fn)
            if dropout > 0.0:
                layers.append(nn.Dropout2d(dropout))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_ch

        self.features = nn.Sequential(*layers)
        # Global pooling to flatten spatial dims
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # --- Classification head: flatten → dense → activation → (dropout) → output
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hid_neurons),
            activation_fn,
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hid_neurons, num_classes)
        )

    def forward(self, x):
        x = self.features(x)    # extract convolutional features
        x = self.pool(x)        # global average pooling
        x = self.classifier(x)  # classification MLP
        return x

#-------------------------TRAIN &  EVAL----------------------

# 1) Helper: one epoch of training
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()    # set model to training mode
    total_loss, correct, count = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="  ▶ Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()             # reset gradients
        outputs = model(imgs)             # forward pass
        loss = criterion(outputs, labels) # compute loss
        loss.backward()                   # backpropagate
        optimizer.step()                  # update weights
        
        # accumulate stats
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        count   += imgs.size(0)
    # compute average loss and accuracy
    avg_loss = total_loss / count
    accuracy = correct / count * 100
    return avg_loss, accuracy

# 2) Helper: one epoch of validation
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

#-------------------TRAINING FUNCTION --------------------

def train_model(
    learning_rate:    float,
    num_filters:      list,
    filter_sizes:     list,
    activation_fn:    str,
    optimiser_fn:     str,
    num_neurons_dense:int,
    weight_decay:     float,
    dropout:          float,
    useBatchNorm:     bool,
    batchSize:        int,
    num_epochs:       int,
    augment: bool,
    dir: str
):
    # a) device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)

    # b) data loaders (stratified 80/20 split + test)
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(dir, num_classes=10, data_augmentation=augment, batch_size=batchSize)


    # d) build model
    activation_dict = {
        "relu":       nn.ReLU(),
        "gelu":       nn.GELU(),
        "elu" : nn.ELU(),
        "mish":       nn.Mish()
    }


    model = SimpleCNN(
        num_classes =10,
        num_filters        = num_filters,
        filter_sizes       = filter_sizes,
        activation_fn      = activation_dict[activation_fn],
        hid_neurons  = num_neurons_dense,
        dropout       = dropout,
        use_batch_norm     = useBatchNorm
    ).to(device)
    model = nn.DataParallel(model)

    # e) loss + optimizer
    loss_fn = nn.CrossEntropyLoss()



    opt_factory = {
        "adam":   lambda: optim.Adam(model.parameters(),   lr=learning_rate, weight_decay=weight_decay),
        "nadam":  lambda: optim.NAdam(model.parameters(),  lr=learning_rate, weight_decay=weight_decay),
        "rmsprop":lambda: optim.RMSprop(model.parameters(),lr=learning_rate, weight_decay=weight_decay),
        "sgd":    lambda: optim.SGD(model.parameters(),    lr=learning_rate, weight_decay=weight_decay),
    }
    optimizer = opt_factory.get(optimiser_fn.lower(), opt_factory["adam"])()



    # f) training + validation
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss,   val_acc   = validate(      model,   val_loader,   loss_fn, device)

        print(f"  ▶ Train → Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  ▶   Val → Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        wandb.log({
            "epoch":       epoch,
            "train_loss":  train_loss,
            "train_acc":   train_acc,
            "val_loss":    val_loss,
            "val_acc":     val_acc
        })

    # g) final test evaluation
    test_loss, test_acc = validate(model, test_loader, loss_fn, device)
    print(f"\n🎯 Test  → Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
    wandb.log({"test_loss": test_loss, "test_acc": test_acc})

    return model



sweep_config = {
    'method': 'bayes',
    'name' : 'cnn-train',
    'metric': {
      'name': 'val_acc',
      'goal': 'maximize'
    },
    'parameters': {
        'num_filters': {
          'values': [[128,128,64,64,32],[16,32,64,128,256]]
        },
        'filter_sizes': {
          'values': [[3,3,3,3,3], [3,5,5,7,7]]
        },
        'weight_decay': {
            'value': 0
        },
        'dropout': {
            'value': 0
        },
        'learning_rate': {
            'values': [1e-4, 1e-3]
        },
        'activation': {
            'values': ['gelu', 'relu']
        },
        'optimiser': {
            'values': ['nadam', 'adam']
        },
        'batch_norm':{
            'value': True
        },
        'augment' : {
             'value': True
        },
        'batch_size': {
            'value': 64
        },
        'dense_layer':{
            'values': [ 256, 512]
        },
        'epoch':{
            'values': [ 10, 20]
        }
    }
}



def main():
    run = wandb.init()

    config = wandb.config

    run_name = f"opt={config.optimiser}_act={config.activation}_nfilt={config.num_filters}_bs={config.batch_size}_layer={config.dense_layer}_dropout={config.dropout}_augment={config.augment}_norm={config.batch_norm}_lr={config.learning_rate}_epoch={config.epoch}"

    # Set the run name
    run.name = run_name


    # train model
    train_model(learning_rate = config.learning_rate,
                num_filters = config.num_filters,
                filter_sizes = config.filter_sizes,
                activation_fn = config.activation,
                optimiser_fn = config.optimiser,
                num_neurons_dense = config.dense_layer,
                weight_decay = config.weight_decay,
                dropout = config.dropout,
                useBatchNorm = config.batch_norm,
                batchSize = config.batch_size,
                num_epochs = config.epoch,
                augment=config.augment,
                dir= directory)




def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN with customizable hyperparameters")
    parser.add_argument("-dr", "--dir",type=str, default="./inaturalist_12K/")
    parser.add_argument("-wp", "--wandb_project", type=str, default="DA6401-A2", help="Weights & Biases project name")
    parser.add_argument("-we", "--wandb_entity", type=str, default="cs24m033-iit-madras", help="Weights & Biases entity")
    parser.add_argument("-e", "--epoch", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--num_filters", type=str, default="[128,128,64,64,32]",
                        help="List of filters per convolutional layer (as stringified list, e.g. '[32,64,64,128,128]')")
    
    parser.add_argument("--filter_sizes", type=str, default="[3,5,5,7,7]",
                        help="List of filter sizes (as stringified list, e.g. '[3,5,5,7,7]')")

    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for optimizer")

    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")

    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")

    parser.add_argument("--activation", type=str, choices=["relu", "gelu", "silu", "mish"], default="gelu",
                        help="Activation function")

    parser.add_argument("--optimiser", type=str, choices=["adam", "nadam", "rmsprop"], default="adam",
                        help="Optimizer type")

    parser.add_argument("--batch_norm", action="store_true", help="Enable batch normalization")

    parser.add_argument("--no_batch_norm", dest="batch_norm", action="store_false", help="Disable batch normalization")
    parser.set_defaults(batch_norm=True)

    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")

    parser.add_argument("--no_augment", dest="augment", action="store_false", help="Disable data augmentation")
    parser.set_defaults(augment=True)

    parser.add_argument("--batch_size", type=int, choices=[32, 64], default=64, help="Batch size")

    parser.add_argument("--dense_layer", type=int, choices=[128, 256, 512], default=512, help="Number of neurons in dense layer")
    return parser.parse_args()


def create_sweep_config(args):
    return {
        "method": "random",
        "name": "cnn_sweep",
        "metric": {
            "goal": "maximize",
            "name": "val_acc"
        },
        "parameters": {
            "num_filters": {"value": eval(args.num_filters)},
            "filter_sizes": {"value": eval(args.filter_sizes)},
            "weight_decay": {"value": args.weight_decay},
            "dropout": {"value": args.dropout},
            "learning_rate": {"value": args.learning_rate},
            "activation": {"value": args.activation},
            "optimiser": {"value": args.optimiser},
            "batch_norm": {"value": args.batch_norm},
            "augment": {"value": args.augment},
            "batch_size": {"value": args.batch_size},
            "dense_layer": {"value": args.dense_layer},
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




#wandb.finish()
