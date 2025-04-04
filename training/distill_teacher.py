import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from tqdm import tqdm

from utils import count_parameters

batch_size = 128
epochs = 10
learning_rate = 0.001

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainset = datasets.SVHN(root='../data', split='train', transform=transform, download=True)
    extraset = datasets.SVHN(root='../data', split='extra', download=True, transform=transform)
    combined_trainset = torch.utils.data.ConcatDataset([trainset, extraset])
    trainloader = torch.utils.data.DataLoader(combined_trainset, batch_size=batch_size, shuffle=True)

    testset = datasets.SVHN(root='../data', split='test', transform=transform, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    print('==> Building model..')
    model = models.resnet18(weights=None)
    print(f"Model has {count_parameters(model)} parameters")

    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 10)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch + 1}", unit="batch")):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        accuracy = evaluate(model, testloader, device)
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}, Accuracy: {accuracy}%")

    torch.save(model.state_dict(), "resnet18_svhn.pth")

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
    main()
    