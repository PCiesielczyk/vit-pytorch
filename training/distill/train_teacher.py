import argparse

import torch
import torchvision.models as models
from tqdm import tqdm

from training.utils import count_parameters, get_data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="SVHN")
parser.add_argument('--train_batch', type=int, default=128)
parser.add_argument('--test_batch', type=int, default=128)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)

FLAGS = parser.parse_args()

def main(args):
    train_kwargs = {'batch_size': args.train_batch, 'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch, 'shuffle': True}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_data_loader(
        args, train_kwargs, test_kwargs)

    print('==> Building model..')
    model = models.resnet18(weights=None)
    print(f"Model has {count_parameters(model)} parameters")

    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 10)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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

        accuracy = evaluate(model, test_loader, device)
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, Accuracy: {accuracy}%")

    torch.save(model.state_dict(), f"resnet18_{args.dataset}.pth")

def evaluate(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

if __name__ == '__main__':
    main(FLAGS)
    