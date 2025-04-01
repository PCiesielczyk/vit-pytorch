import matplotlib.pyplot as plt
import torch

checkpoint_path_1 = "vit_CIFAR-100_ps4.pt"
checkpoint_path_2 = "vit_CIFAR-100_ps8.pt"

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
plt.plot(train_loss_steps, train_loss_history_1, label='patch_size_4')
plt.plot(train_loss_steps, train_loss_history_2, label='patch_size_8')
plt.title(f"Train Loss ViT CIFAR-100 (patch size 4 vs 8)")
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("train_loss_plot.png")
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(epochs, test_loss_history_1, label='patch_size_4')
plt.plot(epochs, test_loss_history_2, label='patch_size_8')
plt.title(f"Test Loss ViT CIFAR-100 (patch size 4 vs 8)")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("test_loss_plot.png")
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(epochs, accuracy_history_1, label='patch_size_4')
plt.plot(epochs, accuracy_history_2, label='patch_size_8')
plt.title(f"Accuracy Over Epochs ViT CIFAR-100 (patch size 4 vs 8)")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_plot.png")
plt.show()
