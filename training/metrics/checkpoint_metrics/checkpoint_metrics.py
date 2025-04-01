import re

import matplotlib.pyplot as plt
import torch


def extract_model_dataset(filename):
    match = re.search(r'([^_]+)_([^_]+)', filename)
    if match:
        return f"{match.group(1).replace('vit', 'ViT')} {match.group(2)}"
    return None

checkpoint_path = "vit_CIFAR-100_ps8.pt"
model_and_dataset = extract_model_dataset(checkpoint_path)

checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=torch.device('cpu'))

train_loss_history = checkpoint['train_loss']
test_loss_history = checkpoint['test_loss']
accuracy_history = checkpoint['accuracy']

train_loss_steps = range(1, len(train_loss_history) + 1)
epochs = range(1, len(test_loss_history) + 1)

plt.figure(figsize=(12, 6))
plt.plot(train_loss_steps, train_loss_history, label='Train Loss', alpha=0.7, color='blue')
plt.title(f"Train Loss {model_and_dataset}")
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("train_loss_plot.png")
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(epochs, test_loss_history, label='Test Loss', alpha=0.8, color='red')
plt.title(f"Test Loss {model_and_dataset}")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("test_loss_plot.png")
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(epochs, accuracy_history, label='Accuracy', alpha=0.8, color='green')
plt.title(f"Accuracy Over Epochs {model_and_dataset}")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_plot.png")
plt.show()
