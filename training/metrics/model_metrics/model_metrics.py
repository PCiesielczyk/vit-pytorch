import pandas as pd
import matplotlib.pyplot as plt


def load_and_process_data(file_path):
    df = pd.read_csv(file_path)
    avg_images_per_second = df["Images Per Second (Per Thread)"].mean()
    avg_epoch_time = df["Epoch Time (s)"].mean()
    last_core_hours = df["Core Hours"].iloc[-1]
    last_accuracy = df["Accuracy"].iloc[-1]
    last_macs = df["MACs"].iloc[-1]
    last_params = df["Params"].iloc[-1]

    return {
        "Images Per Second": avg_images_per_second,
        "Epoch Time (s)": avg_epoch_time,
        "Core Hours": last_core_hours,
        "Top-1 Accuracy": last_accuracy,
        "MACs": last_macs,
        "Params": last_params
    }


data_4x4 = load_and_process_data("vit_CIFAR-10_ps4.csv")
data_8x8 = load_and_process_data("vit_CIFAR-10_ps8.csv")

metrics = ["Images Per Second", "Epoch Time (s)", "Core Hours", "Top-1 Accuracy", "MACs", "Params"]

for metric in metrics:
    plt.figure()
    plt.bar(["Patch 4x4", "Patch 8x8"], [data_4x4[metric], data_8x8[metric]], color=['blue', 'orange'])
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.title(f"Comparison of {metric} for Patch Sizes 4x4 and 8x8")
    plt.savefig(f"{metric.replace(' ', '_').lower()}_comparison.png")
    plt.close()
