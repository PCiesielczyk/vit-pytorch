import matplotlib.pyplot as plt
import torch

checkpoint_path_1 = "CIFAR-10_ViT_Small.pt"
checkpoint_path_2 = "Distill_Small_CIFAR-10_T5_A0_3__soft.pt"
variant = "Distill T=5, alpha=0.3, soft labels"

checkpoint_1 = torch.load(checkpoint_path_1, weights_only=False, map_location=torch.device('cpu'))
checkpoint_2 = torch.load(checkpoint_path_2, weights_only=False, map_location=torch.device('cpu'))

train_loss_history_1 = checkpoint_1['train_loss']
test_loss_history_1 = checkpoint_1['test_loss']
accuracy_history_1 = checkpoint_1['accuracy']

train_loss_history_2 = checkpoint_2['train_loss']
test_loss_history_2 = checkpoint_2['test_loss']
accuracy_history_2 = checkpoint_2['accuracy']

train_loss_steps = range(1, len(train_loss_history_1) + 1)
epochs = range(1, len(test_loss_history_1) + 1)

plt.figure(figsize=(12, 6))
plt.plot(train_loss_steps, train_loss_history_1, label='ViT')
plt.plot(train_loss_steps, train_loss_history_2, label=variant)
plt.title(f"Train Loss ViT Small vs {variant} CIFAR-10")
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("train_loss_plot.png")
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(epochs, test_loss_history_1, label='ViT')
plt.plot(epochs, test_loss_history_2, label=variant)
plt.title(f"Test Loss ViT Small vs {variant} CIFAR-10")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("test_loss_plot.png")
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(epochs, accuracy_history_1, label='ViT')
plt.plot(epochs, accuracy_history_2, label=variant)
plt.title(f"Accuracy Over Epochs ViT Small vs {variant} CIFAR-10")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_plot.png")
plt.show()
