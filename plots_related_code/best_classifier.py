import matplotlib.pyplot as plt
import numpy as np

# Define log file path
log_file_path = "Top3.txt"  # Update with the correct path if needed

# Define model names
models = ["RandomForest_Without_Fisher", "RandomForest_With_Fisher", "DeepLearning_Without_Fisher"]
train_accs = {model: [] for model in models}
test_accs = {model: [] for model in models}

# Read and clean the log file
with open(log_file_path, "r", encoding="utf-8") as file:
    lines = [line.replace("�", "").strip() for line in file.readlines()]

current_model = None

for i, line in enumerate(lines):
    # Identify which model is currently being processed
    if "========== Running RandomForest_Without_Fisher ==========" in line:
        current_model = "RandomForest_Without_Fisher"
    elif "========== Running RandomForest_With_Fisher ==========" in line:
        current_model = "RandomForest_With_Fisher"
    elif "========== Running DeepLearning_Without_Fisher ==========" in line:
        current_model = "DeepLearning_Without_Fisher"

    # Extract Train and Test Accuracy
    elif current_model and "Train Acc:" in line:
        train_acc_parts = line.split(":")[1].strip().split()
        train_acc_mean = float(train_acc_parts[0])  # First part is mean
        train_acc_std = float(train_acc_parts[1])   # Second part is std

        test_line = lines[i + 1].strip()  # The next line should contain the Test Acc
        test_acc_parts = test_line.split(":")[1].strip().split()
        test_acc_mean = float(test_acc_parts[0])  # First part is mean
        test_acc_std = float(test_acc_parts[1])   # Second part is std

        train_accs[current_model] = [train_acc_mean, train_acc_std]
        test_accs[current_model] = [test_acc_mean, test_acc_std]

# Compute overall mean and std for all models
avg_train_acc_mean = np.mean([train_accs[model][0] for model in models])
avg_train_acc_std = np.mean([train_accs[model][1] for model in models])
avg_test_acc_mean = np.mean([test_accs[model][0] for model in models])
avg_test_acc_std = np.mean([test_accs[model][1] for model in models])

# Create bar chart
x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))

rects1 = ax.bar(x - width/2, [train_accs[model][0] for model in models], width, 
                yerr=[train_accs[model][1] for model in models], label='Train Accuracy', capsize=5)
rects2 = ax.bar(x + width/2, [test_accs[model][0] for model in models], width, 
                yerr=[test_accs[model][1] for model in models], label='Test Accuracy', capsize=5)

# Labels and title
ax.set_xlabel('Models')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Mean Accuracy with Standard Deviation for Selected Models')
ax.set_xticks(x)
ax.set_xticklabels(["RF w/o Fisher", "RF w/ Fisher", "DL w/o Fisher"])
ax.legend()

# Save the figure
fig.savefig("best_classifier.png", dpi=600, bbox_inches='tight')

# Show the plot
plt.tight_layout()
plt.show()

print("Figure saved as 'accuracy_comparison.png'")
