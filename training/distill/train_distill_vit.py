# https://github.com/BoyuanJackChen/MiniProject2_VisTrans.git

import argparse
import csv
import os
import pprint
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import optim, nn
from torchvision.models import resnet18

from training.utils import get_data_loader, count_parameters, calculate_macs
from vit_pytorch.distill import DistillableViT, DistillWrapper

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="SVHN")
parser.add_argument('--transform', type=str, default="None")
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--checkpoint', type=int, default=100)
parser.add_argument('--load_checkpoint', type=str, default=None)
parser.add_argument('--resume', '-r', action='store_true')

# General
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--train_batch', type=int, default=10)
parser.add_argument('--test_batch', type=int, default=100)

# ViT
parser.add_argument('--dimhead', default="64", type=int)
parser.add_argument('--heads', default="8", type=int)
parser.add_argument('--depth', default="6", type=int)
parser.add_argument('--mlp_dim', default="512", type=int)

# Distill
parser.add_argument('--teacher_weights', type=str, default=None)
parser.add_argument('--temperature', type=float, default=3.0)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--hard', type=bool, default=False)

FLAGS = parser.parse_args()


def main(args):
    # Use gpu if available
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(42)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device is:", device)
    print(f"Running on device: {device}")
    # Parameters
    train_kwargs = {'batch_size': args.train_batch, 'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch, 'shuffle': True}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Checkpoint saving and loading
    PATH = "../checkpoint/"
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    # Function from utils. Normalization is implemented
    train_loader, test_loader = get_data_loader(
        args, train_kwargs, test_kwargs)

    print('==> Loading Dataset..')
    # patch_size is the number of pixels for each patch's width and height. Not patch number.
    if args.dataset == "CIFAR-10":
        image_size = 32
        patch_size = 4
        num_classes = 10
    elif args.dataset == "CIFAR-100":
        image_size = 32
        patch_size = 4
        num_classes = 100
    elif args.dataset == "MNIST" or args.dataset == "FashionMNIST":
        image_size = 28
        patch_size = 7
        num_classes = 10
    elif args.dataset == "ImageNet_1k":
        image_size = 224
        patch_size = 56
        num_classes = 1000
    elif args.dataset == "SVHN":
        image_size = 32
        patch_size = 4
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print('==> Building model..')

    teacher = resnet18()
    teacher.fc = nn.Linear(teacher.fc.in_features, num_classes)
    state_dict = torch.load(args.teacher_weights, device)
    teacher.load_state_dict(state_dict)

    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()

    dim = int(args.dimhead)
    heads = int(args.heads)
    depth = int(args.depth)
    mlp_dim = int(args.mlp_dim)

    student_model = DistillableViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=dim,
                                   depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=0.1, emb_dropout=0.1)

    distiller = DistillWrapper(student=student_model, teacher=teacher, temperature=args.temperature, alpha=args.alpha,
                               hard=args.hard)

    print(f"Model has {count_parameters(distiller)} parameters")
    print(
        f"Distillation with {distiller.alpha} alpha, {distiller.temperature} temperature and {'hard' if args.hard else 'soft'} labels")
    print(f"Student model has {count_parameters(student_model)} parameters")

    if device == 'cuda':
        distiller = torch.nn.DataParallel(distiller)  # make parallel
        cudnn.benchmark = True

    distiller = distiller.to(device)
    optimizer = optim.Adam(distiller.parameters(), lr=args.lr)

    macs, params = calculate_macs(student_model, input_size=(3, image_size, image_size))

    start_epoch = 1
    train_loss_history, test_loss_history, test_accuracy_history = np.array(
        []), np.array([]), np.array([])

    metrics = {"examples_seen": 0, "total_time": 0, "img_per_sec": 0, "core_hours": 0, "macs": macs,
               "params": params}

    print('==> Training starts')
    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()
        print('Epoch:', epoch)
        train_loss_history = train_epoch(distiller, device, optimizer, train_loader, train_loss_history, metrics)
        test_loss_history, test_accuracy_history = evaluate_model(distiller, device, test_loader, test_loss_history,
                                                                  test_accuracy_history, metrics)

        epoch_time = time.time() - start_time

        data = {
            "Epoch": epoch,
            "Examples Seen": metrics["examples_seen"],
            "Images Per Second (Per Thread)": metrics["img_per_sec"] / torch.get_num_threads(),
            "Core Hours": metrics["core_hours"],
            "Epoch Time (s)": epoch_time,
            "Accuracy": metrics["accuracy"],
            "MACs": metrics["macs"],
            "Params": metrics["params"]
        }

        print(f"Epoch took: {epoch_time:.2f} seconds")
        pprint.pprint(data)

        with open(f"Distill_{args.dataset}_training_metrics.csv", mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(data)

        if epoch % args.checkpoint == 0 or epoch == args.epochs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': distiller.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss_history,
                'test_loss': test_loss_history,
                'accuracy': test_accuracy_history,
            }, PATH + f"/Distill_{args.dataset}_e{epoch}_b{args.train_batch}_lr{args.lr}.pt")
            print(f"Checkpoint {args.dataset}_e{epoch}_b{args.train_batch}_lr{args.lr}.pt saved")


def train_epoch(distiller, device, optimizer, data_loader, loss_history, metrics):
    total_samples = len(data_loader.dataset)
    distiller.train()

    start_time = time.perf_counter()
    examples_seen = 0

    for i, (data, target) in enumerate(data_loader):
        batch_size = data.size(0)
        examples_seen += batch_size

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = distiller(data, target)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            elapsed_time = time.perf_counter() - start_time
            img_per_sec = examples_seen / elapsed_time

            print('[' + '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()) + ', Img/sec: {:.2f}'.format(img_per_sec))
            loss_history = np.append(loss_history, loss.item())

    epoch_time = time.perf_counter() - start_time
    metrics["examples_seen"] += examples_seen
    metrics["total_time"] += epoch_time
    metrics["img_per_sec"] = metrics["examples_seen"] / metrics["total_time"]
    metrics["core_hours"] = (metrics["total_time"] * torch.get_num_threads()) / 3600

    return loss_history


def evaluate_model(distiller, device, data_loader, loss_history, accuracy_history, metrics):
    distiller.eval()
    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            output = distiller.student(data)
            loss = F.cross_entropy(output, target)
            _, pred = torch.max(output, dim=1)
            total_loss += loss.item()
            correct_samples += pred.eq(target).sum().item()

    avg_loss = total_loss / total_samples
    loss_history = np.append(loss_history, avg_loss)
    accuracy = correct_samples / total_samples
    accuracy_history = np.append(accuracy_history, accuracy)
    metrics["accuracy"] = accuracy
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')

    return loss_history, accuracy_history


if __name__ == "__main__":
    main(FLAGS)
