import numpy as np
from sklearn.datasets import load_iris
import tkinter as tk


# Load and prepare the Iris dataset
def load_data():
    iris = load_iris()
    data = iris.data
    target = iris.target

    # Filter only Versicolor (label 1) and Virginica (label 2)
    mask = (target == 1) | (target == 2)
    data = data[mask]
    target = target[mask]

    # Use only the second and third features
    data = data[:, 1:3]  # Petal width and petal length
    target = target - 1  # Change labels to 0 (Versicolor) and 1 (Virginica)
    return data, target


# Compute entropy
def compute_entropy(target):
    # Calculate probabilities of each class
    class_probs = np.bincount(target) / len(target)

    # Avoid log(0) by adding a small epsilon value
    entropy = -np.sum(class_probs * np.log2(class_probs + 1e-6))
    return entropy


# Compute classification error
def compute_error(predictions, true_labels):
    return np.mean(predictions != true_labels)


# Generate all possible thresholds for each feature
def generate_thresholds(data):
    thresholds = {}
    for i in range(data.shape[1]):
        unique_values = np.unique(data[:, i])
        thresholds[i] = (unique_values[:-1] + unique_values[1:]) / 2  # Midpoints
    return thresholds


# Predict labels for a given split
def predict(tree, data):
    predictions = []
    for point in data:
        node = tree
        while "label" not in node:
            feature, threshold = node["split_feature"], node["threshold"]
            if point[feature] <= threshold:
                node = node["left"]
            else:
                node = node["right"]
        predictions.append(node["label"])
    return np.array(predictions)
def generate_tree_entropy(data, target, thresholds, level, max_level):
    if level == max_level or len(np.unique(target)) == 1:
        # Stop splitting if max depth is reached or data is pure
        return {"label": np.round(np.mean(target))}

    best_tree = None
    best_entropy = float("inf")

    for feature, thres_values in thresholds.items():
        for threshold in thres_values:
            # Split data
            left_mask = data[:, feature] <= threshold
            right_mask = ~left_mask

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            left_data, left_target = data[left_mask], target[left_mask]
            right_data, right_target = data[right_mask], target[right_mask]

            # Check if any side has only one unique label (if yes, don't split)
            if len(np.unique(left_target)) == 1 or len(np.unique(right_target)) == 1:
                continue  # Skip this split if one of the children has only one label

            # Compute entropy before and after the split
            entropy_before = compute_entropy(target)
            entropy_left = compute_entropy(left_target)
            entropy_right = compute_entropy(right_target)

            # Weighted sum of entropy after split
            entropy_after = (len(left_target) / len(target)) * entropy_left + (len(right_target) / len(target)) * entropy_right

            # Choose the split that minimizes entropy
            if entropy_after < best_entropy:
                best_entropy = entropy_after
                # Generate left and right subtrees
                left_tree = generate_tree_entropy(left_data, left_target, thresholds, level + 1, max_level)
                right_tree = generate_tree_entropy(right_data, right_target, thresholds, level + 1, max_level)

                # Combine into a tree
                best_tree = {
                    "split_feature": feature,
                    "threshold": threshold,
                    "left": left_tree,
                    "right": right_tree,
                }

    # If no valid split found, return a leaf node with the label
    if best_tree is None:
        return {"label": np.round(np.mean(target))}

    return best_tree



# Main function to build and evaluate the entropy-based decision tree
def entropy_tree(data, target, k):
    thresholds = generate_thresholds(data)
    best_tree = generate_tree_entropy(data, target, thresholds, level=0, max_level=k)
    predictions = predict(best_tree, data)
    error = compute_error(predictions, target)
    return best_tree, error


# Visualize the tree in GUI using tkinter
def draw_tree(canvas, tree, x, y, x_offset, y_offset, depth=0):
    if "label" in tree:
        canvas.create_text(x, y, text=f"Label: {tree['label']}", fill="blue")
        return

    feature, threshold = tree["split_feature"], tree["threshold"]
    canvas.create_text(x, y, text=f"F{feature} ≤ {threshold}", fill="black")

    # Calculate positions for left and right branches
    left_x = x - x_offset // (2 ** depth)
    right_x = x + x_offset // (2 ** depth)
    next_y = y + y_offset

    # Draw lines to branches
    canvas.create_line(x, y, left_x, next_y, fill="gray")
    canvas.create_line(x, y, right_x, next_y, fill="gray")

    # Draw left and right subtrees
    draw_tree(canvas, tree["left"], left_x, next_y, x_offset, y_offset, depth + 1)
    draw_tree(canvas, tree["right"], right_x, next_y, x_offset, y_offset, depth + 1)


def visualize_tree_gui(tree):
    window = tk.Tk()
    window.title("Decision Tree Visualization")

    canvas_width = 800
    canvas_height = 600
    canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg="white")
    canvas.pack()

    # Start drawing tree in the center
    draw_tree(canvas, tree, canvas_width // 2, 50, canvas_width // 4, 100)

    window.mainloop()


# Run the algorithm
if __name__ == "__main__":
    # Load data
    data, target = load_data()

    # Set maximum depth
    k = 3

    # Build and evaluate entropy-based decision tree
    tree, error = entropy_tree(data, target, k)

    # Print results
    print("Entropy-Based Decision Tree:")
    print(tree)
    visualize_tree_gui(tree)
    print(f"Classification Error: {error:.4f}")
