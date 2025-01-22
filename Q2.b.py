import numpy as np
import Utils


# Compute entropy

def compute_entropy(target):
    if len(target) == 0:
        return 0
    counts = np.bincount(target.astype(int))
    probabilities = counts / len(target)
    # Remove zero probabilities before computing log
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))


# Generate all possible thresholds for each feature
def generate_thresholds(data):
    thresholds = {}
    for i in range(data.shape[1]):
        unique_values = np.unique(data[:, i])
        thresholds[i] = (unique_values[:-1] + unique_values[1:]) / 2  # Midpoints
    return thresholds


def generate_tree_entropy(data, target, thresholds, level, max_level):
    # Base case: if all samples have same label
    unique_labels = np.unique(target)
    if len(unique_labels) == 1:
        return {"label": unique_labels[0]}

    # If max depth reached
    if level == max_level:
        return {"label": np.round(np.mean(target))}

    current_entropy = compute_entropy(target)
    best_tree = None
    best_gain = 0
    min_gain = 0.1  # Increased minimum gain threshold

    for feature, thres_values in thresholds.items():
        for threshold in thres_values:
            left_mask = data[:, feature] <= threshold
            right_mask = ~left_mask

            # Skip if either split is too small
            if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
                continue

            left_target = target[left_mask]
            right_target = target[right_mask]

            # Skip if both splits would result in the same label
            left_unique = np.unique(left_target)
            right_unique = np.unique(right_target)

            # Critical check: Skip if this split would lead to redundant leaf nodes
            if (len(left_unique) == 1 and len(right_unique) == 1 and
                    left_unique[0] == right_unique[0]):
                continue

            # Calculate information gain
            left_entropy = compute_entropy(left_target)
            right_entropy = compute_entropy(right_target)
            weighted_entropy = (len(left_target) * left_entropy +
                                len(right_target) * right_entropy) / len(target)
            gain = current_entropy - weighted_entropy

            # Only split if we get meaningful information gain
            if gain > min_gain and gain > best_gain:
                best_gain = gain
                # Before creating split, check if resulting nodes would be pure
                left_tree = ({"label": left_target[0]} if len(np.unique(left_target)) == 1
                             else generate_tree_entropy(data[left_mask], left_target,
                                                        thresholds, level + 1, max_level))
                right_tree = ({"label": right_target[0]} if len(np.unique(right_target)) == 1
                              else generate_tree_entropy(data[right_mask], right_target,
                                                         thresholds, level + 1, max_level))

                # Only create split if children are different
                if not (isinstance(left_tree, dict) and isinstance(right_tree, dict) and
                        'label' in left_tree and 'label' in right_tree and
                        left_tree['label'] == right_tree['label']):
                    best_tree = {
                        "split_feature": feature,
                        "threshold": threshold,
                        "left": left_tree,
                        "right": right_tree
                    }

    # If no good split found
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
    k = 2

    # Build and evaluate entropy-based decision tree
    tree, error = entropy_tree(data, target, k)

    # Print results
    print("Entropy-Based Decision Tree:")
    print(tree)
    Utils.visualize_tree_gui(tree, "Entropy-Based Decision Tree")
    print(f"Classification Error: {error:.4f}")
