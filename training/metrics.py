import torch
import matplotlib.pyplot as plt

checkpoint_path = "vit_SVHN_e10_b10_lr0.0001.pt"
metrics_path = "vit_SVHN_training_metrics.csv"

checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=torch.device('cpu'))

train_loss_history = checkpoint['train_loss']
test_loss_history = checkpoint['test_loss']
accuracy_history = checkpoint['accuracy']

train_loss_steps = range(1, len(train_loss_history) + 1)
epochs = range(1, len(test_loss_history) + 1)

plt.figure(figsize=(12, 6))
plt.plot(train_loss_steps, train_loss_history, label='Train Loss', marker='o', linestyle='--', alpha=0.7, color='blue')
plt.title("Train Loss")
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("train_loss_plot.png")
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(epochs, test_loss_history, label='Test Loss', marker='o', linestyle='-', alpha=0.8, color='red')
plt.title("Test Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("test_loss_plot.png")
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(epochs, accuracy_history, label='Accuracy', marker='o', linestyle='-', alpha=0.8, color='green')
plt.title("Accuracy Over Epochs")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_plot.png")
plt.show()
