import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Define the directory containing log files
log_dir = os.path.dirname(os.path.abspath(__file__))

# Regular expressions to extract Deep Learning parameters and Traditional Classifier parameters
dl_params_pattern = r"Hidden Dim:\s+(\d+).*?Num Layers:\s+(\d+).*?Batch Size:\s+(\d+).*?Learning Rate:\s+([\d.e-]+)"
rf_params_pattern = r"n_estimators:\s+(\d+)"

# Regular expression to extract the average performance metrics
avg_performance_pattern = r"(DEEP_LEARNING|DEEP_LEARNING_LDA|TRADITIONAL|TRADITIONAL_LDA)\s+[\d.]+\s+([\d.]+)"

# Data storage for parameters and performance metrics
dl_data_subjects = []
dl_data_datasets = []
rf_data_subjects = []
rf_data_datasets = []

# Get all log files from the directory
log_files = [f for f in os.listdir(log_dir) if f.endswith(".txt")]

# Read and process each log file
for file_name in log_files:
    file_path = os.path.join(log_dir, file_name)

    with open(file_path, "r") as f:
        content = f.read()

    # Extract deep learning parameters
    dl_params = re.search(dl_params_pattern, content, re.DOTALL)
    if dl_params:
        hidden_dim, num_layers, batch_size, learning_rate = dl_params.groups()
        hidden_dim, num_layers, batch_size = map(int, [hidden_dim, num_layers, batch_size])
        learning_rate = float(learning_rate)

    # Extract traditional classifier parameters
    rf_params = re.search(rf_params_pattern, content)
    if rf_params:
        n_estimators = int(rf_params.group(1))

    # Extract performance averages for all subjects
    match_subjects = re.search(r"===== Average Performance Across All Subjects =====\n(.*?)\n===== Average Performance Across All Datasets =====", content, re.DOTALL)
    if match_subjects:
        avg_performance_subjects = re.findall(avg_performance_pattern, match_subjects.group(1))
        for model, test_acc in avg_performance_subjects:
            test_acc = float(test_acc)
            if "DEEP_LEARNING" in model:
                dl_data_subjects.append((hidden_dim, num_layers, batch_size, learning_rate, model, test_acc))
            else:
                rf_data_subjects.append((n_estimators, model, test_acc))

    # Extract performance averages for all datasets
    match_datasets = re.search(r"===== Average Performance Across All Datasets =====\n(.*)", content, re.DOTALL)
    if match_datasets:
        avg_performance_datasets = re.findall(avg_performance_pattern, match_datasets.group(1))
        for model, test_acc in avg_performance_datasets:
            test_acc = float(test_acc)
            if "DEEP_LEARNING" in model:
                dl_data_datasets.append((hidden_dim, num_layers, batch_size, learning_rate, model, test_acc))
            else:
                rf_data_datasets.append((n_estimators, model, test_acc))

# Convert to DataFrames
df_dl_subjects = pd.DataFrame(dl_data_subjects, columns=["Hidden Dim", "Num Layers", "Batch Size", "Learning Rate", "Type", "Test Acc"])
df_dl_datasets = pd.DataFrame(dl_data_datasets, columns=["Hidden Dim", "Num Layers", "Batch Size", "Learning Rate", "Type", "Test Acc"])
df_rf_subjects = pd.DataFrame(rf_data_subjects, columns=["n_estimators", "Type", "Test Acc"])
df_rf_datasets = pd.DataFrame(rf_data_datasets, columns=["n_estimators", "Type", "Test Acc"])

# Function to create grouped bar plots for Deep Learning parameters
def plot_dl_variation(df, parameter, xlabel, title_suffix):
    if df.empty:
        print(f"Skipping {parameter} - No data available for {title_suffix}")
        return
    
    avg_df = df.groupby([parameter, "Type"])["Test Acc"].mean().unstack()
    
    if not avg_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        avg_df.plot(kind="bar", ax=ax, width=0.6, color=["blue", "orange"])  # Colors for DL and DL_LDA

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Average Test Accuracy (%)")
        ax.set_title(f"Effect of {xlabel} on Deep Learning Accuracy - {title_suffix}")
        plt.xticks(rotation=0)
        plt.grid()
        plt.legend(title="Type")

        # Save the figure
        filename = f"DL_{parameter}_{title_suffix.replace(' ', '_')}.png"
        plt.savefig(os.path.join(log_dir, filename))
        plt.close()
        print(f"Saved: {filename}")

# Generate grouped bar plots for each Deep Learning parameter
for param, xlabel in [("Hidden Dim", "Hidden Dimensions"), ("Num Layers", "Number of Layers"),
                      ("Batch Size", "Batch Size"), ("Learning Rate", "Learning Rate")]:
    plot_dl_variation(df_dl_subjects, param, xlabel, "Subjects")
    plot_dl_variation(df_dl_datasets, param, xlabel, "Datasets")

# Function to create grouped bar plots for Random Forest parameter variation
def plot_rf_variation(df, title_suffix):
    if df.empty:
        print(f"Skipping n_estimators - No data available for {title_suffix}")
        return
    
    avg_df = df.groupby(["n_estimators", "Type"])["Test Acc"].mean().unstack()

    if not avg_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        avg_df.plot(kind="bar", ax=ax, width=0.6, color=["green", "red"])  # Colors for Traditional and Traditional_LDA

        ax.set_xlabel("Number of Estimators")
        ax.set_ylabel("Average Test Accuracy (%)")
        ax.set_title(f"Effect of n_estimators on Random Forest Accuracy - {title_suffix}")
        plt.xticks(rotation=0)
        plt.grid()
        plt.legend(title="Type")

        # Save the figure
        filename = f"RF_n_estimators_{title_suffix.replace(' ', '_')}.png"
        plt.savefig(os.path.join(log_dir, filename))
        plt.close()
        print(f"Saved: {filename}")

# Generate grouped bar plots for n_estimators variation
plot_rf_variation(df_rf_subjects, "Subjects")
plot_rf_variation(df_rf_datasets, "Datasets")

print("All figures saved successfully!")
