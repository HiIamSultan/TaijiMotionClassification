import re
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# Define the log file path
log_file_path = "Top3.txt"  # Update if necessary

# Initialize a list to store unique feature numbers per array
all_features = []

# Read the log file and extract feature numbers
with open(log_file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

i = 0
while i < len(lines):
    if "Top 32 discriminative features:" in lines[i]:
        feature_set = set()  # Use a set to store unique features per array
        
        # Extract numbers from the current line
        feature_set.update(map(int, re.findall(r"\d+", lines[i])))

        # Ensure we also capture the next line unless "Processing Subject" appears
        if i + 1 < len(lines) and not lines[i + 1].startswith("Processing Subject"):
            feature_set.update(map(int, re.findall(r"\d+", lines[i + 1])))
            i += 1  # Move to the next line, since we have processed it

        # Convert the set to a list and add to the global list
        all_features.extend(feature_set)
    
    i += 1  # Move to the next line

# Count the frequency of each feature
feature_counts = Counter(all_features)

# Initialize 17 blocks for features (0-3, 4-7, ..., 64-67)
num_blocks = 17
block_frequencies = [0] * num_blocks

for feature, count in feature_counts.items():
    block_index = feature // 4  # Compute block index (each block has 4 features)
    if block_index < num_blocks:  # Ensure valid block range
        block_frequencies[block_index] += count  # Sum frequency for each block

# Normalize frequencies (max frequency = 1)
max_freq = max(block_frequencies)
normalized_frequencies = [freq / max_freq for freq in block_frequencies]

# Create block labels (1 to 17)
block_labels = list(range(1, num_blocks + 1))

# Plot the feature blocks
plt.figure(figsize=(10, 5))
plt.plot(block_labels, normalized_frequencies, marker='o', linestyle='-', color='b')
plt.fill_between(block_labels, normalized_frequencies, alpha=0.2, color='blue')

# Labels and title
plt.xlabel("Features")
plt.ylabel("Normalized Frequency (Max = 1)")
plt.title("Normalized Feature Frequency")
plt.xticks(block_labels)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Save the figure
plt.savefig("best_feature.png", dpi=600, bbox_inches="tight")

print("Figure saved as 'feature_frequency_blocks.png'")
