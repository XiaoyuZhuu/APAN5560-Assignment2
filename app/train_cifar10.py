# app/train_cifar10.py
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .cnn import SimpleCNN
from .labels import CIFAR10_CLASSES

def get_dataloaders(batch_size=128, num_workers=2):
    transform_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2470, 0.2435, 0.2616)),
    ])
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    trainset = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=transform_train)
    testset  = datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader  = DataLoader(testset,  batch_size=256, shuffle=False, num_workers=num_workers)
    return trainloader, testloader

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_sum += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return loss_sum / total, correct / total

def train(epochs=10, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    trainloader, testloader = get_dataloaders()
    model = SimpleCNN(num_classes=len(CIFAR10_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    outdir = Path("models")
    outdir.mkdir(exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for i, (images, labels) in enumerate(trainloader, 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
            if i % 100 == 0:
                print(f"Epoch {epoch} Step {i}: loss={running/100:.4f}")
                running = 0.0

        val_loss, val_acc = evaluate(model, testloader, device)
        print(f"[Epoch {epoch}] val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = {
                "model_state": model.state_dict(),
                "classes": CIFAR10_CLASSES,
            }
            torch.save(ckpt, outdir / "cnn_cifar10.pt")
            print(f"Saved best model (acc={best_acc:.4f}) to models/cnn_cifar10.pt")

    print("Training done. Best acc:", best_acc)

if __name__ == "__main__":
    # 你也可以通过环境变量快速改 epoch / lr
    epochs = int(os.getenv("EPOCHS", "10"))
    lr = float(os.getenv("LR", "0.001"))
    train(epochs=epochs, lr=lr)
