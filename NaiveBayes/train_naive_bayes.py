import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sklearn.datasets as datasets
from NaiveBayes import NaiveBayes

X, y = datasets.make_classification(n_samples= 1000, n_features= 10, n_classes=2, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

naive_bayes = NaiveBayes()
naive_bayes.fit(X_train, y_train)
predictions = naive_bayes.predict(X_test)
accuracy = np.sum(predictions == y_test)/len(y_test)
print(accuracy)

# Apply PCA to reduce dimensions for visualization
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Set up the plot style
plt.style.use("seaborn-v0_8-darkgrid")
plt.figure(figsize=(10, 6))

# Plot training data
cmap = plt.get_cmap("summer")
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=cmap, alpha=0.5, s=30, label="Training Data")

# Plot correctly classified points
correct_indices = np.where(predictions == y_test)[0]
plt.scatter(X_test_pca[correct_indices, 0], X_test_pca[correct_indices, 1],
            color="green", s=50, label="Correct Prediction", marker="o")

# Plot incorrectly classified points
incorrect_indices = np.where(predictions != y_test)[0]
plt.scatter(X_test_pca[incorrect_indices, 0], X_test_pca[incorrect_indices, 1],
            color="red", s=50, label="Incorrect Prediction", marker="x")

# Add title and labels
plt.title(f"Naive Bayes Classification (Accuracy: {accuracy:.4f})", fontsize=14)
plt.xlabel("Principal Component 1", fontsize=12)
plt.ylabel("Principal Component 2", fontsize=12)
plt.legend(fontsize=10)


# Optional: Create a meshgrid to visualize decision boundary
def plot_decision_boundary():
    h = 0.02  # Step size in the mesh
    x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
    y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Transform back to original feature space using inverse_transform
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Use the PCA's inverse_transform to go back to original space
    grid_points_original = pca.inverse_transform(grid_points)

    # Get predictions for all grid points
    Z = naive_bayes.predict(grid_points_original)
    Z = np.array(Z).reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.coolwarm)


# Uncomment the line below if you want to show the decision boundary
# (it may be computationally intensive)
# plot_decision_boundary()

# Save the figure
plt.tight_layout()
plt.savefig("../images/naive_bayes_visualization.png", dpi=300, bbox_inches="tight")
plt.show()

# Print accuracy information
print(f"Naive Bayes Accuracy: {accuracy:.4f}")
print(f"Correctly classified samples: {len(correct_indices)}")
print(f"Incorrectly classified samples: {len(incorrect_indices)}")
