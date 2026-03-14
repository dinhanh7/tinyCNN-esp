#!/usr/bin/env python3
"""
train.py

Train the Tiny MobileNet model on CIFAR-10 and save best weights to
tiny_mobilenet_cifar10.pth

Usage: python train.py --epochs 100 --batch-size 128
"""
import argparse
import os
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class InvertedResidual(nn.Module):
    """Inverted Residual block as specified.

    Structure: 1x1 pointwise (expand) -> 3x3 depthwise -> 1x1 pointwise (project)
    Activation: ReLU6 ONLY after expand and depthwise. No activation after final projection.
    BatchNorm after every conv.
    Skip connection when stride == 1 and in_channels == out_channels.
    """

    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = (self.stride == 1 and in_channels == out_channels)

        layers = []
        # Expand: 1x1 pointwise
        if hidden_dim != in_channels:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        else:
            # when expand_ratio == 1, we still treat it as identity for expand
            hidden_dim = in_channels

        # Depthwise 3x3
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        ])

        # Project: 1x1
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            return x + out
        else:
            return out


class TinyMobileNetCIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Conv_Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
        )

        # MBConv blocks as specified
        self.mbconv1 = nn.Sequential(
            InvertedResidual(16, 16, stride=1, expand_ratio=1),
            nn.ReLU6(inplace=True),
        )

        self.mbconv2 = nn.Sequential(
            InvertedResidual(16, 24, stride=2, expand_ratio=2),
            nn.ReLU6(inplace=True),
        )

        self.mbconv3 = nn.Sequential(
            InvertedResidual(24, 32, stride=2, expand_ratio=2),
            nn.ReLU6(inplace=True),
        )

        self.mbconv4 = nn.Sequential(
            InvertedResidual(32, 64, stride=2, expand_ratio=2),
            nn.ReLU6(inplace=True),
        )

        # Conv_Head
        self.conv_head = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),
        )

        # Pool and classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.conv_head(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def get_dataloaders(batch_size, num_workers=4):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=num_workers)

    return trainloader, testloader


def train_one_epoch(model, device, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="train", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += inputs.size(0)
        pbar.set_postfix(loss=running_loss / total, acc=100.0 * correct / total)

    return running_loss / total, 100.0 * correct / total


def evaluate(model, device, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="eval", leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += inputs.size(0)
            pbar.set_postfix(loss=running_loss / total, acc=100.0 * correct / total)

    return running_loss / total, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-path", type=str, default="tiny_mobilenet_cifar10.pth")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    trainloader, testloader = get_dataloaders(args.batch_size, num_workers=args.workers)

    model = TinyMobileNetCIFAR10(num_classes=10)
    
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming carefully from weights: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, device, trainloader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, device, testloader, criterion)
        scheduler.step()

        print(f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | val_loss={val_loss:.4f} val_acc={val_acc:.2f}%")

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved best model (val_acc={best_acc:.2f}%) to {args.save_path}")

        # Simple early stopping target
        if best_acc >= 80.0:
            print("Target accuracy reached. Stopping training.")
            break

    print("Training finished. Best validation accuracy: {:.2f}%".format(best_acc))


if __name__ == "__main__":
    main()
