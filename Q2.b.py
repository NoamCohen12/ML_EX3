import numpy as np
import Utils


# Compute entropy
def compute_entropy(target):
    # Calculate probabilities of each class
    class_probs = np.bincount(target) / len(target)

    # Avoid log(0) by adding a small epsilon value
    entropy = -np.sum(class_probs * np.log2(class_probs + 1e-6))
    return entropy


# Generate all possible thresholds for each feature
def generate_thresholds(data):
    thresholds = {}
    for i in range(data.shape[1]):
        unique_values = np.unique(data[:, i])
        thresholds[i] = (unique_values[:-1] + unique_values[1:]) / 2  # Midpoints
    return thresholds


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
            entropy_after = (len(left_target) / len(target)) * entropy_left + (
                        len(right_target) / len(target)) * entropy_right

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
    predictions = Utils.predict(best_tree, data)
    error = Utils.compute_error(predictions, target)
    return best_tree, error


# Run the algorithm
if __name__ == "__main__":
    # Load data
    data, target = Utils.load_data()

    # Set maximum depth
    k = 3

    # Build and evaluate entropy-based decision tree
    tree, error = entropy_tree(data, target, k)

    # Print results
    print("Entropy-Based Decision Tree:")
    print(tree)
    Utils.visualize_tree_gui(tree, "Entropy-Based Decision Tree")
    print(f"Classification Error: {error:.4f}")
