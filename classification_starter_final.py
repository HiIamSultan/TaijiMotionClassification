'''
Md Sultan Mahmud (mqm7099)
{
    This Python script implements a classification pipeline using both traditional machine learning and 
    deep learning models with Leave-One-Subject-Out (LOSO) cross-validation. It includes feature selection 
    based on ANOVA F-scores, Fisher’s Linear Discriminant Analysis (LDA) for dimensionality reduction, and 
    classification using Random Forest and Multi-Layer Perceptron (MLP) models. The script processes multiple datasets, 
    evaluates classifiers before and after LDA projection, and computes key performance metrics such as accuracy, 
    precision, recall, and F1-score. It also generates confusion matrices and systematically logs results for 
    reproducibility. The script automates classification across datasets and subjects, providing statistical summaries 
    for model evaluation.
}
'''
import sys
import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model_starter import TraditionalClassifier
from model_starter import  MLP
from torch.optim.lr_scheduler import StepLR
from sklearn.feature_selection import f_classif, RFE
from sklearn.ensemble import RandomForestClassifier

def most_discriminative_features(feats, labels, top_k=10):
    """
    Identifies the top-K most discriminative features based on ANOVA F-score.
    """
    f_scores, _ = f_classif(feats, labels)
    sorted_indices = np.argsort(f_scores)[::-1]  # Descending order
    top_features = sorted_indices[:top_k]
    print(f"Top {top_k} discriminative features:", top_features)
    return top_features

def feature_selection(feats, labels, num_features=10):
    """
    Implements feature selection by retaining a subset of the most important features.

    Returns:
        - selected_features (numpy.ndarray): Boolean mask indicating selected features.
    """
    # Get Most Discriminative Individual Features
    selected_feature_subset = most_discriminative_features(feats, labels, num_features)

    return selected_feature_subset

def convert_features_to_loader(train_feats_proj, train_labels, test_feats_proj, test_labels, batch_size=32):
    """
    Converts NumPy arrays to PyTorch tensors and creates DataLoader instances.

    Args:
        train_feats_proj (numpy.ndarray): Training feature set.
        train_labels (numpy.ndarray): Training labels.
        test_feats_proj (numpy.ndarray): Testing feature set.
        test_labels (numpy.ndarray): Testing labels.
        batch_size (int, optional): Batch size for DataLoader. Default is 32.

    Returns:
        tuple: (train_loader, test_loader)
    """

    # Convert training data to PyTorch tensors
    train_feats_tensor = torch.tensor(train_feats_proj, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)

    # Convert testing data to PyTorch tensors
    test_feats_tensor = torch.tensor(test_feats_proj, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

    # Create TensorDataset
    train_dataset = TensorDataset(train_feats_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_feats_tensor, test_labels_tensor)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"DataLoader: Training set - {len(train_loader)} batches, Testing set - {len(test_loader)} batches")

    return train_loader, test_loader

def deep_learning(train_feats_proj, train_labels, test_feats_proj, test_labels, 
                  input_dim=None, output_dim=None, hidden_dim=64, num_layers=2, 
                  batch_size=64, learning_rate=0.0001, epochs=100):
    """
    Trains an MLP classifier and evaluates it on test data using GPU acceleration if available.

    Args:
        train_feats_proj (numpy.ndarray): Training feature set.
        train_labels (numpy.ndarray): Training labels.
        test_feats_proj (numpy.ndarray): Testing feature set.
        test_labels (numpy.ndarray): Testing labels.
        input_dim (int): Number of input features.
        output_dim (int): Number of output classes.
        hidden_dim (int): Number of hidden units per layer.
        num_layers (int): Number of hidden layers.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.

    Returns:
        tuple: (train_accuracy, test_accuracy, precision, recall, f1_score, train_labels_pred, test_labels_pred)
    """

    # Set device: GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")

    if input_dim is None:
        input_dim = train_feats_proj.shape[1]
    if output_dim is None:
        output_dim = len(np.unique(train_labels))

    # Define the MLP model and move it to the selected device
    model = MLP(input_dim, output_dim, hidden_dim, nn.ReLU, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # Convert features into PyTorch DataLoaders and move data to device
    def convert_features_to_loader(train_feats, train_labels, test_feats, test_labels, batch_size=32):
        """
        Converts NumPy arrays to PyTorch tensors and creates DataLoader instances.

        Args:
            train_feats (numpy.ndarray): Training feature set.
            train_labels (numpy.ndarray): Training labels.
            test_feats (numpy.ndarray): Testing feature set.
            test_labels (numpy.ndarray): Testing labels.
            batch_size (int, optional): Batch size for DataLoader. Default is 32.

        Returns:
            tuple: (train_loader, test_loader)
        """
        # Convert to PyTorch tensors and move to device
        train_feats_tensor = torch.tensor(train_feats, dtype=torch.float32).to(device)
        train_labels_tensor = torch.tensor(train_labels, dtype=torch.long).to(device)

        test_feats_tensor = torch.tensor(test_feats, dtype=torch.float32).to(device)
        test_labels_tensor = torch.tensor(test_labels, dtype=torch.long).to(device)

        # Create TensorDataset
        train_dataset = TensorDataset(train_feats_tensor, train_labels_tensor)
        test_dataset = TensorDataset(test_feats_tensor, test_labels_tensor)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # print(f"DataLoader: Training set - {len(train_loader)} batches, Testing set - {len(test_loader)} batches")
        return train_loader, test_loader

    # Prepare data loaders
    train_loader, test_loader = convert_features_to_loader(train_feats_proj, train_labels, test_feats_proj, test_labels, batch_size)

    # Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)  # Move to GPU
            optimizer.zero_grad()
            outputs = model(batch_data)  
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    # Function to evaluate performance
    def evaluate(loader):
        all_preds, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for batch_data, batch_labels in loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)  # Move to GPU
                outputs = model(batch_data)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())  # Move back to CPU
                all_labels.extend(batch_labels.cpu().numpy())
        return np.array(all_labels), np.array(all_preds)

    # Compute performance metrics
    train_labels_actual, train_labels_pred = evaluate(train_loader)
    test_labels_actual, test_labels_pred = evaluate(test_loader)

    train_acc = accuracy_score(train_labels_actual, train_labels_pred) * 100
    test_acc = accuracy_score(test_labels_actual, test_labels_pred) * 100
    precision = precision_score(test_labels_actual, test_labels_pred, average="weighted")
    recall = recall_score(test_labels_actual, test_labels_pred, average="weighted")
    f1 = f1_score(test_labels_actual, test_labels_pred, average="weighted")

    print(f"Train Accuracy: {train_acc:.2f}% | Test Accuracy: {test_acc:.2f}%")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")

    return train_acc, test_acc, precision, recall, f1, train_labels_pred, test_labels_pred

def perform_traditional(train_feats_proj, train_labels, test_feats_proj, test_labels, 
                        n_estimators=150, random_state=42):
    """
    Trains and evaluates a traditional classifier with specified parameters.

    Args:
        train_feats_proj (numpy.ndarray): Training features.
        train_labels (numpy.ndarray): Training labels.
        test_feats_proj (numpy.ndarray): Testing features.
        test_labels (numpy.ndarray): Testing labels.
        n_estimators (int): Number of estimators for RandomForestClassifier.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Evaluation metrics and predictions.
    """

    clf = TraditionalClassifier(n_estimators=n_estimators, random_state=random_state)  # Use parameters
    clf.fit(train_feats_proj, train_labels)

    pred_train_labels = clf.predict(train_feats_proj)
    pred_test_labels = clf.predict(test_feats_proj)

    train_acc = accuracy_score(train_labels, pred_train_labels) * 100
    test_acc = accuracy_score(test_labels, pred_test_labels) * 100
    precision = precision_score(test_labels, pred_test_labels, average="weighted")
    recall = recall_score(test_labels, pred_test_labels, average="weighted")
    f1 = f1_score(test_labels, pred_test_labels, average="weighted")

    print(f"Train Accuracy: {train_acc:.2f}% | Test Accuracy: {test_acc:.2f}%")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")

    return train_acc, test_acc, precision, recall, f1, pred_train_labels, pred_test_labels


def load_new_dataset(dataset_name="Taiji_dataset_200.csv", subject_index=9, verbose=True):
    """
    Loads dataset, applies feature selection, and performs Leave-One-Subject-Out (LOSO) cross-validation.

    Args:
        dataset_name (str): Path to the dataset file (supports Taiji_dataset_100, 200, and 300).
        subject_index (int): Index of the subject to be left out for LOSO.
        verbose (bool): Whether to print dataset details.

    Returns:
        tuple: train_feats, train_labels, test_feats, test_labels
    """

    # Load dataset from the specified file
    dataset = np.loadtxt(dataset_name, delimiter=",", dtype=float, skiprows=1, usecols=range(0, 70))

    # Extract `person_idxs` (last column) and `labels` (second last column)
    person_idxs = dataset[:, -1]  # All rows, last column (Subject ID)
    labels = dataset[:, -2]  # All rows, second last column (Class Labels)
    feats = dataset[:, :-2]  # All rows, all columns except the last two (Features)

    # Feature Selection (Remove features with zero variance)
    feature_mask = np.var(feats, axis=0) > 0  # Keep features with non-zero variance
    feats = feats[:, feature_mask]

    # Implement Leave-One-Subject-Out (LOSO) split
    train_mask = person_idxs != subject_index
    test_mask = person_idxs == subject_index

    train_feats = feats[train_mask]
    train_labels = labels[train_mask].astype(int)
    test_feats = feats[test_mask]
    test_labels = labels[test_mask].astype(int)

    # Apply Feature Selection (train first, then align test)
    selected_features = feature_selection(train_feats, train_labels, num_features=32)
    train_feats = train_feats[:, selected_features]
    test_feats = test_feats[:, selected_features]  # Use the same feature mask for test data

    # Print dataset details if verbose is enabled
    if verbose:
        print(f"\nDataset Loaded: {dataset_name}")
        print(f"\t# of Classes: {len(np.unique(train_labels))}")
        print(f"\t# of Features after Selection: {train_feats.shape[1]}")
        print(f"\t# of Training Samples: {train_feats.shape[0]}")
        # print("\t# per Class in Train Dataset:")
        # for cls in np.unique(train_labels):
        #     print(f"\t\tClass {cls}: {np.sum(train_labels == cls)}")
        print(f"\t# of Testing Samples: {test_feats.shape[0]}")
        # print("\t# per Class in Test Dataset:")
        # for cls in np.unique(test_labels):
        #     print(f"\t\tClass {cls}: {np.sum(test_labels == cls)}")

    return train_feats, train_labels, test_feats, test_labels

def plot_conf_mats(dataset, model_name, subject_idx, train_labels, pred_train_labels, test_labels, pred_test_labels):
    """
    Plots and saves confusion matrices per subject.

    Args:
        dataset (str): Dataset name.
        model_name (str): Model name (e.g., "Traditional_Before_LDA").
        subject_idx (int): Subject number for LOSO.
        train_labels (array): True training labels.
        pred_train_labels (array): Predicted training labels.
        test_labels (array): True testing labels.
        pred_test_labels (array): Predicted testing labels.
    """
    # Ensure that all label sets are the same for the confusion matrix
    unique_labels = np.union1d(np.unique(train_labels), np.unique(test_labels))

    # Compute Confusion Matrices
    train_confusion = confusion_matrix(train_labels, pred_train_labels, labels=unique_labels)
    test_confusion = confusion_matrix(test_labels, pred_test_labels, labels=unique_labels)

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # Training Confusion Matrix
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=train_confusion, display_labels=unique_labels)
    disp.plot(ax=ax, xticks_rotation='vertical')
    ax.set_title(f'Training Confusion Matrix - {model_name} (Subject {subject_idx})')
    plt.tight_layout()
    plt.savefig(f'results/{dataset}_{model_name}_train_confusion_subject_{subject_idx}.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # Testing Confusion Matrix
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=test_confusion, display_labels=unique_labels)
    disp.plot(ax=ax, xticks_rotation='vertical')
    ax.set_title(f'Testing Confusion Matrix - {model_name} (Subject {subject_idx})')
    plt.tight_layout()
    plt.savefig(f'results/{dataset}_{model_name}_test_confusion_subject_{subject_idx}.png', bbox_inches='tight', pad_inches=0)
    plt.close()


def example_classification():
    """
    An example of performing classification. Except you will need to first project the data.
    """
    train_feats, train_labels, test_feats, test_labels = load_new_dataset()

    # Example Linear Discriminant Analysis classifier with sklearn's implemenetation
    clf = LinearDiscriminantAnalysis()
    clf.fit(train_feats, train_labels)

    # Predict the labels of the training and testing data
    pred_train_labels = clf.predict(train_feats)
    pred_test_labels = clf.predict(test_feats)

    # Get statistics
    plot_conf_mats(train_labels=train_labels, pred_train_labels=pred_train_labels, test_labels=test_labels,
                   pred_test_labels=pred_test_labels)

def fisher_projection(train_feats, train_labels):
    """
    Implements Fisher's Linear Discriminant Analysis (LDA) for dimensionality reduction.

    Args:
        train_feats (numpy.ndarray): Feature matrix (num_samples, num_features).
        train_labels (numpy.ndarray): Training labels (num_samples, ).

    Returns:
        numpy.ndarray: Projection matrix.
    """

    # Compute the overall mean of the training features
    mean_overall = np.mean(train_feats, axis=0)

    # Calculate the mean vector for each class
    class_labels = np.unique(train_labels)
    mean_vectors = {cls: np.mean(train_feats[train_labels == cls], axis=0) for cls in class_labels}

    # Compute the within-class scatter matrix (Sw)
    num_features = train_feats.shape[1]
    Sw = np.zeros((num_features, num_features))

    for cls in class_labels:
        class_feats = train_feats[train_labels == cls]
        mean_diff = class_feats - mean_vectors[cls]  # Deviation from class mean
        Sw += np.dot(mean_diff.T, mean_diff)  # Accumulate scatter for each class

    # Compute the between-class scatter matrix (Sb)
    Sb = np.zeros((num_features, num_features))
    
    for cls in class_labels:
        n_cls = train_feats[train_labels == cls].shape[0]  # Number of samples in the class
        mean_diff = (mean_vectors[cls] - mean_overall).reshape(-1, 1)  # Column vector
        Sb += n_cls * np.dot(mean_diff, mean_diff.T)  # Weighted scatter contribution

    # Compute the transformation matrix J(W) = Sw^-1 * Sb
    # Ensure numerical stability by adding a small identity matrix for regularization
    Sw += np.eye(Sw.shape[0]) * 1e-6  # Regularization to avoid singularity
    J = np.linalg.pinv(Sw) @ Sb  # Compute the pseudo-inverse of Sw and multiply by Sb

    # Solve for the eigenvalues and eigenvectors of J(W)
    eig_vals, eig_vecs = np.linalg.eigh(J)  # eigh ensures real eigenvalues/eigenvectors

    # Sort eigenvectors in descending order based on their absolute eigenvalues
    sorted_indices = np.argsort(-np.abs(eig_vals))
    eig_vecs = eig_vecs[:, sorted_indices]

    # Select the top two eigenvectors for dimensionality reduction
    num_components = min(32, len(class_labels) - 1)  # LDA allows at most (C-1) components
    print(f"Number of LDA components selected: {num_components}")  # Print num_components
    lda_projection_matrix = eig_vecs[:, :num_components]

    # Return the selected eigenvectors
    return lda_projection_matrix

def classification(datasets=["Taiji_dataset_100.csv", "Taiji_dataset_200.csv", "Taiji_dataset_300.csv"], 
                   hidden_dim=64, num_layers=2, batch_size=32, learning_rate=0.001, epochs=20,
                   n_estimators=150, random_state=42):
    """
    Implements LOSO cross-validation for the selected top 3 classifiers:
    1. Random Forest (w/o Fisher Projection)
    2. Random Forest (w/ Fisher Projection)
    3. Deep Learning (w/o Fisher Projection)

    Reports:
    - LOSO results (10 results, mean, std)
    - Per-class results (39 results, mean, std)
    - Dataset-level mean and std
    - Confusion Matrices (10 per model)
    """

    unique_subjects = np.arange(1, 11)

    classifiers = {
        "RandomForest_Without_Fisher": lambda train_feats, train_labels, test_feats, test_labels: 
            perform_traditional(train_feats, train_labels, test_feats, test_labels, n_estimators, random_state),
        "RandomForest_With_Fisher": lambda train_feats, train_labels, test_feats, test_labels: 
            perform_traditional(np.dot(train_feats, fisher_projection(train_feats, train_labels)), train_labels,
                                np.dot(test_feats, fisher_projection(train_feats, train_labels)), test_labels,
                                n_estimators, random_state),
        "DeepLearning_Without_Fisher": lambda train_feats, train_labels, test_feats, test_labels:
            deep_learning(train_feats, train_labels, test_feats, test_labels,
                          input_dim=train_feats.shape[1], output_dim=len(np.unique(train_labels)),
                          hidden_dim=hidden_dim, num_layers=num_layers, batch_size=batch_size,
                          learning_rate=learning_rate, epochs=epochs),
    }

    for classifier_name, classifier_func in classifiers.items():
        print(f"\n========== Running {classifier_name} ==========")
        
        dataset_results = {dataset: [] for dataset in datasets}
        loso_results = {subj: [] for subj in unique_subjects}
        class_results = {cls: [] for cls in range(39)}

        for dataset_name in datasets:
            print(f"\nProcessing Dataset: {dataset_name}")

            for subject_idx in unique_subjects:
                print(f"\nProcessing Subject {subject_idx}...")

                # Load dataset with the current subject left out
                train_feats, train_labels, test_feats, test_labels = load_new_dataset(
                    subject_index=subject_idx, dataset_name=dataset_name
                )

                # Train and evaluate classifier
                train_acc, test_acc, precision, recall, f1, pred_train_labels, pred_test_labels = classifier_func(
                    train_feats, train_labels, test_feats, test_labels
                )

                # Store results per subject (LOSO)
                loso_results[subject_idx].append((train_acc, test_acc, precision, recall, f1))

                # Store dataset-specific results
                dataset_results[dataset_name].append((train_acc, test_acc, precision, recall, f1))

                # Store per-class results
                unique_test_classes = np.unique(test_labels)
                for cls in unique_test_classes:
                    cls_mask = test_labels == cls
                    cls_acc = accuracy_score(test_labels[cls_mask], pred_test_labels[cls_mask]) * 100
                    class_results[cls].append(cls_acc)

                # Generate and save confusion matrices
                print("\nGenerating Confusion Matrix...")
                plot_conf_mats(dataset=dataset_name, model_name=classifier_name, subject_idx=subject_idx,
                               train_labels=train_labels, pred_train_labels=pred_train_labels,
                               test_labels=test_labels, pred_test_labels=pred_test_labels)

        # Function to compute mean and std
        def compute_avg_metrics(metrics):
            metrics_array = np.array(metrics)
            return {
                "Mean": np.mean(metrics_array, axis=0),
                "Std": np.std(metrics_array, axis=0)
            }

        # Print LOSO Results
        print(f"\n===== LOSO Results for {classifier_name} =====")
        print(f"{'Subject':<10}{'Train Acc':<10}{'Test Acc':<10}{'Precision':<10}{'Recall':<10}{'F1-Score':<10}")
        print("=" * 70)

        loso_means = []
        for subject_idx in unique_subjects:
            if loso_results[subject_idx]:
                stats = compute_avg_metrics(loso_results[subject_idx])
                print(f"{subject_idx:<10}{stats['Mean'][0]:<10.2f}{stats['Mean'][1]:<10.2f}"
                      f"{stats['Mean'][2]:<10.4f}{stats['Mean'][3]:<10.4f}{stats['Mean'][4]:<10.4f}")
                loso_means.append(stats["Mean"])

        # Compute overall mean and std for LOSO
        print("\nLOSO Mean ± Std:")
        print(f"Train Acc: {np.mean(loso_means, axis=0)[0]:.2f} ± {np.std(loso_means, axis=0)[0]:.2f}")
        print(f"Test Acc: {np.mean(loso_means, axis=0)[1]:.2f} ± {np.std(loso_means, axis=0)[1]:.2f}")

        # Print All-Class Results
        print(f"\n===== All-Class Results for {classifier_name} =====")
        print(f"{'Class':<10}{'Accuracy':<10}")
        print("=" * 30)

        class_means = []
        for cls in range(39):
            if class_results[cls]:
                cls_mean = np.mean(class_results[cls])
                cls_std = np.std(class_results[cls])
                print(f"{cls:<10}{cls_mean:.2f} ± {cls_std:.2f}")
                class_means.append(cls_mean)

        print("\nAll-Class Mean ± Std:")
        print(f"Accuracy: {np.mean(class_means):.2f} ± {np.std(class_means):.2f}")

        # Print Dataset-Level Results
        print(f"\n===== Dataset-Level Results for {classifier_name} =====")
        for dataset_name in datasets:
            if dataset_results[dataset_name]:
                dataset_stats = compute_avg_metrics(dataset_results[dataset_name])
                print(f"Dataset: {dataset_name}")
                print(f"Train Acc: {dataset_stats['Mean'][0]:.2f} ± {dataset_stats['Std'][0]:.2f}")
                print(f"Test Acc: {dataset_stats['Mean'][1]:.2f} ± {dataset_stats['Std'][1]:.2f}")
                print(f"Precision: {dataset_stats['Mean'][2]:.4f} ± {dataset_stats['Std'][2]:.4f}")
                print(f"Recall: {dataset_stats['Mean'][3]:.4f} ± {dataset_stats['Std'][3]:.4f}")
                print(f"F1-Score: {dataset_stats['Mean'][4]:.4f} ± {dataset_stats['Std'][4]:.4f}")
                print("=" * 70)


class Tee:
    """Class to duplicate stdout/stderr to both terminal and a log file."""
    def __init__(self, filename):
        self.terminal = sys.stdout  # Store original stdout
        self.log = open(filename, "w", buffering=1)  # Enable line buffering

    def write(self, message):
        self.terminal.write(message)  # Write to terminal
        self.terminal.flush()  # Ensure immediate terminal output
        self.log.write(message)  # Write to log file
        self.log.flush()  # Force write to log file

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main():
    # Generate a unique log filename using a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"classification_log_{timestamp}.txt"

    # Redirect stdout and stderr to Tee (both terminal and file)
    sys.stdout = Tee(log_filename)
    sys.stderr = sys.stdout  # Redirect stderr as well

    print(f"Logging terminal output to {log_filename}...\n")

    # Run the classification process
    classification()

    # Reset stdout and stderr to default
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    print(f"\nClassification process completed. Logs are saved in {log_filename}")

if __name__ == '__main__':
    main()

