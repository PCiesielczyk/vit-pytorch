import argparse

import torch
import torchvision.models as models
from torch.nn import Dropout
from tqdm import tqdm

from training.early_stopping import EarlyStopping
from training.utils import count_parameters, get_data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="CIFAR-100")
parser.add_argument('--train_batch', type=int, default=128)
parser.add_argument('--test_batch', type=int, default=128)
parser.add_argument('--epochs', type=int, default=70)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--augmentation', type=bool, default=True)

FLAGS = parser.parse_args()

def main(args):
    train_kwargs = {'batch_size': args.train_batch, 'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch, 'shuffle': True}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_data_loader(
        args, train_kwargs, test_kwargs)

    print('==> Building model..')
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    print(f"Model has {count_parameters(model)} parameters")

    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(
        Dropout(p=0.2),
        torch.nn.Linear(num_ftrs, 100)
    )
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    early_stopping = EarlyStopping(patience=20, verbose=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="batch")):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_loss = 0.0
        accuracy = 0.0
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                accuracy += (predicted == labels).sum().item()

        val_loss /= len(test_loader)
        accuracy = 100.0 * accuracy / len(test_loader.dataset)

        print(
            f"Epoch {epoch + 1}, Train Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

        scheduler.step()

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    torch.save(model.state_dict(), f"resnet18_{args.dataset}.pth")

if __name__ == '__main__':
    main(FLAGS)
    