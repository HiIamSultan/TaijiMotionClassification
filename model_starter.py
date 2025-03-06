from networkx.classes import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class TraditionalClassifier(RandomForestClassifier):
    def __init__(self, class_weight="balanced", n_estimators=150, random_state=42):
        """
        Initializes a RandomForestClassifier with an optional class weighting.
        """
        super().__init__(n_estimators=n_estimators, class_weight=class_weight, random_state=random_state)

    def evaluate(self, train_feats, train_labels, test_feats, test_labels):
        """
        Evaluates the classifier using accuracy, confusion matrix, classification report, and risk-based loss.
        """
        pred_train_labels = self.predict(train_feats)
        pred_test_labels = self.predict(test_feats)

        train_acc = accuracy_score(train_labels, pred_train_labels)
        test_acc = accuracy_score(test_labels, pred_test_labels)

        print(f"\nTraining Accuracy: {train_acc * 100:.2f}%")
        print(f"Testing Accuracy: {test_acc * 100:.2f}%")

        print("\nClassification Report (Test Data):\n", classification_report(test_labels, pred_test_labels))
        print("\nConfusion Matrix (Test Data):\n", confusion_matrix(test_labels, pred_test_labels))

        # Compute risk matrix loss
        risk_matrix = self.get_risk_matrix(num_classes=len(np.unique(test_labels)))
        risk_loss = self.compute_risk_loss(test_labels, pred_test_labels, risk_matrix)

        print(f"\nTotal Expected Risk-Based Loss: {risk_loss:.2f}")

    def get_risk_matrix(self, num_classes):
        """
        Creates a risk matrix where misclassification incurs different penalties.
        """
        risk_matrix = np.ones((num_classes, num_classes)) * 2  # Default penalty is 2
        np.fill_diagonal(risk_matrix, 0)  # No penalty for correct predictions

        # Example: Assign higher penalties for certain misclassifications (adjust as needed)
        for i in range(num_classes):
            for j in range(num_classes):
                if abs(i - j) == 1:  # If classes are adjacent, lower penalty
                    risk_matrix[i, j] = 1  
                elif abs(i - j) >= 3:  # If classes are far apart, higher penalty
                    risk_matrix[i, j] = 3  

        return risk_matrix

    def compute_risk_loss(self, true_labels, pred_labels, risk_matrix):
        """
        Computes the total risk-based loss based on misclassifications.
        """
        total_loss = 0
        for i in range(len(true_labels)):
            total_loss += risk_matrix[true_labels[i], pred_labels[i]]

        return total_loss / len(true_labels)  # Normalize by number of samples


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, activation=nn.ReLU, num_layers=2):
        super(MLP, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
