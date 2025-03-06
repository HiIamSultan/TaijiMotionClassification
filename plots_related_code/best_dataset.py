import matplotlib.pyplot as plt
import numpy as np
import re

# Define log file path
log_file_path = "Top3.txt"

# Define dataset names
datasets = ["Taiji_dataset_100.csv", "Taiji_dataset_200.csv", "Taiji_dataset_300.csv"]
dataset_accs = {dataset: {"train": [], "test": []} for dataset in datasets}

# Read and clean the log file
with open(log_file_path, "r", encoding="utf-8") as file:
    lines = [line.replace("�", "").strip() for line in file.readlines()]

current_dataset = None

for i, line in enumerate(lines):
    # Identify which dataset is currently being processed
    for dataset in datasets:
        if f"Dataset: {dataset}" in line:
            current_dataset = dataset

    # Extract Train and Test Accuracy
    if current_dataset and "Train Acc:" in line:
        train_acc_parts = line.split(":")[1].strip().split()
        train_acc_mean = float(train_acc_parts[0])  # First part is mean
        train_acc_std = float(train_acc_parts[1])   # Second part is std

        test_line = lines[i + 1].strip()  # The next line should contain the Test Acc
        test_acc_parts = test_line.split(":")[1].strip().split()
        test_acc_mean = float(test_acc_parts[0])  # First part is mean
        test_acc_std = float(test_acc_parts[1])   # Second part is std

        dataset_accs[current_dataset]["train"].append((train_acc_mean, train_acc_std))
        dataset_accs[current_dataset]["test"].append((test_acc_mean, test_acc_std))

# Compute overall mean and std for each dataset across all models
avg_dataset_accs = {dataset: {"train_mean": 0, "train_std": 0, "test_mean": 0, "test_std": 0} for dataset in datasets}

for dataset in datasets:
    train_means = [entry[0] for entry in dataset_accs[dataset]["train"]]
    train_stds = [entry[1] for entry in dataset_accs[dataset]["train"]]
    test_means = [entry[0] for entry in dataset_accs[dataset]["test"]]
    test_stds = [entry[1] for entry in dataset_accs[dataset]["test"]]

    avg_dataset_accs[dataset]["train_mean"] = np.mean(train_means)
    avg_dataset_accs[dataset]["train_std"] = np.mean(train_stds)
    avg_dataset_accs[dataset]["test_mean"] = np.mean(test_means)
    avg_dataset_accs[dataset]["test_std"] = np.mean(test_stds)

# Sort datasets based on descending test accuracy
sorted_datasets = sorted(datasets, key=lambda d: avg_dataset_accs[d]["test_mean"], reverse=True)

# Prepare sorted data
sorted_train_means = [avg_dataset_accs[d]["train_mean"] for d in sorted_datasets]
sorted_train_stds = [avg_dataset_accs[d]["train_std"] for d in sorted_datasets]

sorted_test_means = [avg_dataset_accs[d]["test_mean"] for d in sorted_datasets]
sorted_test_stds = [avg_dataset_accs[d]["test_std"] for d in sorted_datasets]

# Create grouped bar chart for train and test accuracy
x = np.arange(len(sorted_datasets))
width = 0.35  # Width of bars

fig, ax = plt.subplots(figsize=(8, 6))

# Train Accuracy Data
ax.bar(x - width/2, sorted_train_means, width, yerr=sorted_train_stds, capsize=5, label='Train Accuracy')

# Test Accuracy Data
ax.bar(x + width/2, sorted_test_means, width, yerr=sorted_test_stds, capsize=5, label='Test Accuracy')

# Labels and title
ax.set_xlabel('Datasets')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Mean Train & Test Accuracy with Standard Deviation (Sorted by Test Accuracy)')
ax.set_xticks(x)
ax.set_xticklabels([d.replace("Taiji_dataset_", "Taiji-").replace(".csv", "") for d in sorted_datasets])
ax.legend()

# Save the figure
fig.savefig("best_dataset.png", dpi=600, bbox_inches='tight')

print("Figure saved as 'sorted_dataset_accuracy.png'")
