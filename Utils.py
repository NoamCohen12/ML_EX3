from sklearn.datasets import load_iris
import tkinter as tk
import numpy as np


# Q2_a.py
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


# Visualize the tree in GUI using tkinter
def draw_tree(canvas, tree, x, y, x_offset, y_offset, title, depth=0):
    # Add the title at the top of the canvas (only once at root level)
    if depth == 0:  # Draw the title only once at the root level
        canvas.create_text(canvas.winfo_width() // 2, 20, text=title, font=("Arial", 16, "bold"), fill="black")

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
    draw_tree(canvas, tree["left"], left_x, next_y, x_offset, y_offset, title, depth + 1)
    draw_tree(canvas, tree["right"], right_x, next_y, x_offset, y_offset, title, depth + 1)








def visualize_tree_gui(tree, title):
    window = tk.Tk()
    window.title("Decision Tree Visualization")

    canvas_width = 800
    canvas_height = 600
    canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg="white")
    canvas.pack()

    # Start drawing tree in the center
    draw_tree(canvas, tree, canvas_width // 2, 50, canvas_width // 4, 100, title=title)

    window.mainloop()

    # Predict labels for a given split0


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


# Compute classification error
def compute_error(predictions, true_labels):
    return np.mean(predictions != true_labels)
