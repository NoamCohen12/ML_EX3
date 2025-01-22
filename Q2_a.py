import numpy as np
import Utils  # ייבוא המודול






# Generate all possible thresholds for each feature
def generate_thresholds(data):
    thresholds = {}
    for i in range(data.shape[1]):
        unique_values = np.unique(data[:, i])
        thresholds[i] = (unique_values[:-1] + unique_values[1:]) / 2  # Midpoints
    return thresholds


# Recursively generate all trees up to level k
def generate_all_trees(data, target, thresholds, level, max_level):
    if level == max_level or len(np.unique(target)) == 1:
        # Stop splitting if max depth is reached or data is pure
        return {"label": np.round(np.mean(target))}

    best_tree = None
    best_error = float("inf")

    for feature, thres_values in thresholds.items():
        for threshold in thres_values:
            # Split data
            left_mask = data[:, feature] <= threshold
            right_mask = ~left_mask

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            left_data, left_target = data[left_mask], target[left_mask]
            right_data, right_target = data[right_mask], target[right_mask]

            # Generate left and right subtrees
            left_tree = generate_all_trees(left_data, left_target, thresholds, level + 1, max_level)
            right_tree = generate_all_trees(right_data, right_target, thresholds, level + 1, max_level)

            # Combine into a tree
            tree = {
                "split_feature": feature,
                "threshold": threshold,
                "left": left_tree,
                "right": right_tree,
            }

            # Compute error for this tree
            predictions = Utils.predict(tree, data)
            error = Utils.compute_error(predictions, target)

            if error < best_error:
                best_error = error
                best_tree = tree

    return best_tree


# Main function to build and evaluate the brute-force decision tree
def brute_force_tree(data, target, k):
    thresholds = generate_thresholds(data)
    best_tree = generate_all_trees(data, target, thresholds, level=0, max_level=k)
    predictions = Utils.predict(best_tree, data)
    error = Utils.compute_error(predictions, target)
    return best_tree, error


# Run the algorithm
if __name__ == "__main__":
    # Load data
    data, target = Utils.load_data()

    # Set maximum depth
    k = 2

    # Build and evaluate brute-force decision tree
    tree, error = brute_force_tree(data, target, k)

    # Print results
    print("Brute-Force Decision Tree:")
    # print the tree values:
    print(tree)
    Utils.visualize_tree_gui(tree, "Brute-Force Decision Tree")
    print(f"Classification Error: {error:.4f}")
