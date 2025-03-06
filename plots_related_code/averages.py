import os
import re
import numpy as np
import pandas as pd

# Define the directory containing log files
log_dir = os.path.dirname(os.path.abspath(__file__))

# Regular expressions to extract performance metrics
subject_accuracy_pattern = r"(\d+)\s+(traditional|deep_learning)(_LDA)?_metrics\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
dataset_accuracy_pattern = r"(\S+?)\s+(traditional|deep_learning)(_LDA)?_metrics\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"

# Function to compute averages and update log files
def process_log_file(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    # Extract subject-wise performance metrics
    subject_data = []
    subject_matches = re.findall(subject_accuracy_pattern, content)
    for _, model_type, lda_flag, train_acc, test_acc, precision, recall, f1_score in subject_matches:
        model = f"{model_type.upper()}{'_LDA' if lda_flag else ''}"
        subject_data.append((model, float(train_acc), float(test_acc), float(precision), float(recall), float(f1_score)))

    # Extract dataset-wise performance metrics
    dataset_data = []
    dataset_matches = re.findall(dataset_accuracy_pattern, content)
    for dataset, model_type, lda_flag, train_acc, test_acc, precision, recall, f1_score in dataset_matches:
        model = f"{model_type.upper()}{'_LDA' if lda_flag else ''}"
        dataset_data.append((model, float(train_acc), float(test_acc), float(precision), float(recall), float(f1_score)))

    # Convert to DataFrames
    df_subjects = pd.DataFrame(subject_data, columns=["Model", "Train Acc", "Test Acc", "Precision", "Recall", "F1-Score"])
    df_datasets = pd.DataFrame(dataset_data, columns=["Model", "Train Acc", "Test Acc", "Precision", "Recall", "F1-Score"])

    # Calculate averages for each model type
    df_subjects_avg = df_subjects.groupby("Model").mean().reset_index()
    df_datasets_avg = df_datasets.groupby("Model").mean().reset_index()

    # Format the averages as a string for writing back to the log file
    avg_results = "\n===== Average Performance Across All Subjects =====\n"
    avg_results += df_subjects_avg.to_string(index=False) + "\n"
    
    avg_results += "\n===== Average Performance Across All Datasets =====\n"
    avg_results += df_datasets_avg.to_string(index=False) + "\n"

    # Append results to the log file
    with open(file_path, "a") as f:
        f.write(avg_results)

# Get all log files from the directory and process each
log_files = [f for f in os.listdir(log_dir) if f.endswith(".txt")]

for log_file in log_files:
    file_path = os.path.join(log_dir, log_file)
    process_log_file(file_path)

print("Averages computed and appended to all log files successfully!")
