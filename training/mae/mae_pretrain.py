import argparse
import csv
import os
import pprint
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import optim, nn

from training.utils import get_data_loader, count_parameters, calculate_macs
from vit_pytorch import ViT
from vit_pytorch.mae import MAE

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="MAE Pre-training Script")

# Model Encoder Args
parser.add_argument('--encoder_model', type=str, default='vit', choices=['vit', 't2t'])
parser.add_argument('--dim', type=int, default=64)
parser.add_argument('--depth', type=int, default=6)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--mlp_dim', type=int, default=512)

# MAE Specific Args
parser.add_argument('--masking_ratio', type=float, default=0.75)
parser.add_argument('--decoder_dim', type=int, default=32)
parser.add_argument('--decoder_depth', type=int, default=4)
parser.add_argument('--decoder_heads', type=int, default=4)

# Dataset Args
parser.add_argument('--dataset', type=str, default="CIFAR-10")

# Training Args
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', default=1.5e-4, type=float)
parser.add_argument('--weight_decay', default=0.05, type=float)
parser.add_argument('--train_batch', type=int, default=10)
parser.add_argument('--test_batch', type=int, default=100)
parser.add_argument('--checkpoint_interval', type=int, default=10)
parser.add_argument('--output_dir', type=str, default='../checkpoint/mae_pretrain')

FLAGS = parser.parse_args()

def train_mae_epoch(model, device, optimizer, data_loader, metrics, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()

    start_time = time.perf_counter()
    examples_seen = 0

    for i, (data, _) in enumerate(data_loader):
        batch_size = data.size(0)
        examples_seen += batch_size

        data = data.to(device)
        optimizer.zero_grad()
        loss = model(data)

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

def evaluate_mae_epoch(model, device, data_loader, metrics, loss_history):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            loss = model(data)
            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / total_batches
    metrics["loss"] = avg_loss
    loss_history = np.append(loss_history, avg_loss)
    print(f'\nAverage Validation Reconstruction Loss: {avg_loss:.4f}\n')

    return loss_history

def main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    if use_cuda:
        print(f"Available GPUs: {torch.cuda.device_count()}")
    torch.manual_seed(42)
    if use_cuda:
        cudnn.benchmark = True

    CHECKPOINT_DIR = os.path.join(args.output_dir, f"{args.encoder_model}_{args.dataset}")
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    print(f"Checkpoints and logs will be saved in: {CHECKPOINT_DIR}")

    train_kwargs = {'batch_size': args.train_batch, 'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch, 'shuffle': False}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    train_loader, test_loader = get_data_loader(args, train_kwargs, test_kwargs)

    if args.dataset == "CIFAR-10" or args.dataset == "SVHN":
        image_size = 32
        patch_size = 4
        num_classes = 10
        channels = 3
    elif args.dataset == "CIFAR-100":
        image_size = 32
        patch_size = 4
        num_classes = 100
        channels = 3
    elif args.dataset == "ImageNet_1k":
        image_size = 224
        patch_size = 56
        num_classes = 1000
        channels = 3
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Dataset: {args.dataset}, Image Size: {image_size}, Patch Size: {patch_size}, Channels: {channels}")

    print(f"Building Encoder: {args.encoder_model}")
    if args.encoder_model == 'vit':
        encoder = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            mlp_dim=args.mlp_dim,
            channels=channels,
            dropout=0.1,
            emb_dropout=0.1
        )
    else:
        raise ValueError(f"Unsupported encoder model: {args.encoder_model}")

    print("Building MAE Model...")
    mae_model = MAE(
        encoder=encoder,
        masking_ratio=args.masking_ratio,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        decoder_heads=args.decoder_heads,
    ).to(device)

    print(f"MAE Model Parameters: {count_parameters(mae_model)/1e6:.2f} M")

    optimizer = optim.AdamW(mae_model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    macs, params = calculate_macs(mae_model, input_size=(3, image_size, image_size))

    if use_cuda and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        mae_model = nn.DataParallel(mae_model)

    start_epoch = 1

    train_loss_history, test_loss_history = np.array([]), np.array([])
    metrics = {"examples_seen": 0, "total_time": 0, "img_per_sec": 0, "core_hours": 0, "macs": macs,
               "params": params}

    log_filename = f"mae_pretrain_{args.encoder_model}_{args.dataset}_log.csv"
    log_filepath = os.path.join(CHECKPOINT_DIR, log_filename)

    print("==> Starting MAE Pre-training <==")

    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()
        print('Epoch:', epoch)
        train_loss_history = train_mae_epoch(mae_model, device, optimizer, train_loader, metrics, train_loss_history)
        test_loss_history = evaluate_mae_epoch(mae_model, device, test_loader, metrics, test_loss_history)

        epoch_time = time.time() - start_time

        data = {
            "Epoch": epoch,
            "Examples Seen": metrics["examples_seen"],
            "Images Per Second (Per Thread)": metrics["img_per_sec"] / torch.get_num_threads(),
            "Core Hours": metrics["core_hours"],
            "Epoch Time (s)": epoch_time,
            "Loss": metrics["loss"],
            "MACs": metrics["macs"],
            "Params": metrics["params"]
        }

        print(f"Epoch took: {epoch_time:.2f} seconds")
        pprint.pprint(data)

        with open(log_filepath, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(data)

        if epoch % args.checkpoint_interval == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"mae_pretrain_{args.encoder_model}_{args.dataset}_e{epoch}.pt")
            vit_trained_path = os.path.join(CHECKPOINT_DIR, f"vit_pretrained_{args.dataset}_e{epoch}.pt")
            try:
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': mae_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss_history,
                    'test_loss': test_loss_history
                }
                torch.save(save_dict, checkpoint_path)
                torch.save(encoder.state_dict(), vit_trained_path)
                print(f"Checkpoint saved: {checkpoint_path}")
            except Exception as e:
                 print(f"Error saving checkpoint: {e}")

    print(f"\n==> MAE Pre-training Finished. Total time: {metrics['total_time']/3600:.2f} hours <==")
    print(f"Final checkpoints saved in: {CHECKPOINT_DIR}")

if __name__ == "__main__":
    main(FLAGS)
