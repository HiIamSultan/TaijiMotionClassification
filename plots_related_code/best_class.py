import matplotlib.pyplot as plt
import numpy as np
import re

# Define log file path
log_file_path = "Top3.txt"

# Initialize dictionary to store extracted values for all classes
class_accs = {}

# Read the log file
with open(log_file_path, "r", encoding="utf-8") as file:
    lines = [line.replace("�", "").strip() for line in file.readlines()]

current_section = None

for line in lines:
    # Identify the relevant section in the log file
    if "===== All-Class Results for RandomForest_Without_Fisher =====" in line:
        current_section = "RandomForest_Without_Fisher"
    elif "===== All-Class Results for RandomForest_With_Fisher =====" in line:
        current_section = "RandomForest_With_Fisher"
    elif "===== All-Class Results for DeepLearning_Without_Fisher =====" in line:
        current_section = "DeepLearning_Without_Fisher"
    elif "All-Class Mean" in line:  # End of all-class results section
        current_section = None

    # Extract class-wise accuracy
    elif current_section:
        # Use regex to extract numbers from the line
        numbers = re.findall(r"\d+\.\d+|\d+", line)

        if len(numbers) >= 3:
            try:
                class_num = int(numbers[0])  # Extract class number
                class_mean = float(numbers[1])  # Extract mean accuracy
                class_std = float(numbers[2])  # Extract standard deviation

                # Ignore classes with unrealistic accuracy values
                if 0 <= class_mean <= 100:
                    # Dynamically create dictionary keys if class_num is new
                    if class_num not in class_accs:
                        class_accs[class_num] = []
                    class_accs[class_num].append((class_mean, class_std))
            except ValueError:
                print(f"Skipping malformed line: {line}")  # Debugging message

# Compute overall mean and std for each class across all models
avg_class_accs = {i: {"mean": 0, "std": 0} for i in sorted(class_accs.keys())}

for i in class_accs.keys():
    means = [entry[0] for entry in class_accs[i]]
    stds = [entry[1] for entry in class_accs[i]]

    avg_class_accs[i]["mean"] = np.mean(means)
    avg_class_accs[i]["std"] = np.mean(stds)

# Filtering valid class numbers
valid_classes = [i for i in avg_class_accs.keys() if 0 <= avg_class_accs[i]["mean"] <= 100]

# Sort classes in descending order based on accuracy
sorted_classes = sorted(valid_classes, key=lambda i: avg_class_accs[i]["mean"], reverse=True)

# Create bar chart
x = np.arange(len(sorted_classes))
width = 0.6

fig, ax = plt.subplots(figsize=(12, 6))

class_means = [avg_class_accs[i]["mean"] for i in sorted_classes]
class_stds = [avg_class_accs[i]["std"] for i in sorted_classes]

rects = ax.bar(x, class_means, width, yerr=class_stds, capsize=5, label='Class Accuracy')

# Labels and title
ax.set_xlabel('Class Number')
ax.set_ylabel('Test Accuracy (%)')
ax.set_title('Mean Accuracy with Standard Deviation (Sorted by Test Accuracy)')
ax.set_xticks(x)
ax.set_xticklabels(sorted_classes, rotation=90)
ax.legend()

# Save the figure
fig.savefig("best_class.png", dpi=600, bbox_inches='tight')

print("Sorted figure saved as 'sorted_class_accuracy_comparison.png'")
