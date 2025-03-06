import matplotlib.pyplot as plt
import numpy as np
import re

# Define log file path
log_file_path = "Top3.txt"

# Initialize dictionary to store extracted values for 10 subjects
subject_train_accs = {i: [] for i in range(1, 11)}
subject_test_accs = {i: [] for i in range(1, 11)}

# Read the log file
with open(log_file_path, "r", encoding="utf-8") as file:
    lines = [line.replace("�", "").strip() for line in file.readlines()]

current_section = None

for line in lines:
    # Identify the relevant section in the log file
    if "===== LOSO Results for RandomForest_Without_Fisher =====" in line:
        current_section = "RandomForest_Without_Fisher"
    elif "===== LOSO Results for RandomForest_With_Fisher =====" in line:
        current_section = "RandomForest_With_Fisher"
    elif "===== LOSO Results for DeepLearning_Without_Fisher =====" in line:
        current_section = "DeepLearning_Without_Fisher"
    elif "LOSO Mean" in line:  # End of LOSO results section
        current_section = None

    # Extract subject-wise accuracy
    elif current_section:
        numbers = re.findall(r"\d+\.\d+|\d+", line)

        if len(numbers) >= 3:
            try:
                subject_num = int(numbers[0])  # Extract subject number
                train_acc = float(numbers[1])  # Extract train accuracy
                test_acc = float(numbers[2])  # Extract test accuracy

                subject_train_accs[subject_num].append(train_acc)
                subject_test_accs[subject_num].append(test_acc)
            except ValueError:
                print(f"Skipping malformed line: {line}")  # Debugging message

# Compute overall mean and std for each subject across all models
avg_subject_train_accs = {i: {"mean": 0, "std": 0} for i in range(1, 11)}
avg_subject_test_accs = {i: {"mean": 0, "std": 0} for i in range(1, 11)}

for i in range(1, 11):
    train_means = subject_train_accs[i] if subject_train_accs[i] else [0]
    test_means = subject_test_accs[i] if subject_test_accs[i] else [0]

    avg_subject_train_accs[i]["mean"] = np.mean(train_means)
    avg_subject_train_accs[i]["std"] = np.std(train_means)

    avg_subject_test_accs[i]["mean"] = np.mean(test_means)
    avg_subject_test_accs[i]["std"] = np.std(test_means)

# Sort subjects based on descending test accuracy
sorted_subjects = sorted(avg_subject_test_accs.keys(), key=lambda i: avg_subject_test_accs[i]["mean"], reverse=True)

# Prepare sorted data
sorted_train_means = [avg_subject_train_accs[i]["mean"] for i in sorted_subjects]
sorted_train_stds = [avg_subject_train_accs[i]["std"] for i in sorted_subjects]

sorted_test_means = [avg_subject_test_accs[i]["mean"] for i in sorted_subjects]
sorted_test_stds = [avg_subject_test_accs[i]["std"] for i in sorted_subjects]

# Create grouped bar chart for train and test accuracy
x = np.arange(len(sorted_subjects))
width = 0.35  # Width of bars

fig, ax = plt.subplots(figsize=(12, 6))

# Train Accuracy Data
ax.bar(x - width/2, sorted_train_means, width, yerr=sorted_train_stds, capsize=5, label='Train Accuracy')

# Test Accuracy Data
ax.bar(x + width/2, sorted_test_means, width, yerr=sorted_test_stds, capsize=5, label='Test Accuracy')

# Labels and title
ax.set_xlabel('Subjects')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Mean Train & Test Accuracy with Standard Deviation (Sorted by Test Accuracy)')
ax.set_xticks(x)
ax.set_xticklabels(sorted_subjects)  # Use sorted subject numbers
ax.legend()

# Save the figure
fig.savefig("best_subjects.png", dpi=600, bbox_inches='tight')

print("Figure saved as 'sorted_train_test_accuracy_subjects.png'")
