import numpy as np
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
import Utils
from tabulate import tabulate


# Function to implement k-NN prediction
def knn_predict(X_train, y_train, X_test, k, p):
    # Compute distances between test points and training points
    distances = cdist(X_test, X_train, metric='minkowski', p=p)

    # Get the indices of the k nearest neighbors
    nearest_indices = np.argsort(distances, axis=1)[:, :k]

    # Predict the labels based on majority voting
    predictions = []
    for idx in nearest_indices:
        neighbors = y_train[idx]
        majority_class = np.bincount(neighbors).argmax()
        predictions.append(majority_class)

    return np.array(predictions)


# Function to evaluate k-NN for different values of k and p
def evaluate_knn(data, target, k_values, p_values, num_repeats=100):
    results = {}

    for k in k_values:
        for p in p_values:
            train_errors = []
            test_errors = []

            # Repeat the evaluation `num_repeats` times
            for _ in range(num_repeats):
                # Split data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.5, random_state=None)

                # Train and test the model
                y_train_pred = knn_predict(X_train, y_train, X_train, k, p)
                y_test_pred = knn_predict(X_train, y_train, X_test, k, p)

                # Compute errors on training and test sets
                train_error = Utils.compute_error(y_train_pred, y_train)
                test_error = Utils.compute_error(y_test_pred, y_test)

                train_errors.append(train_error)
                test_errors.append(test_error)

            # Compute average errors and the difference between them
            avg_train_error = np.mean(train_errors)
            avg_test_error = np.mean(test_errors)
            error_diff =  avg_test_error - avg_train_error

            # Store the results for this (k, p) pair
            results[(k, p)] = {
                'avg_train_error': avg_train_error,
                'avg_test_error': avg_test_error,
                'error_diff': error_diff
            }

    return results


def print_results(results):
    # יצירת רשימה של שורות עבור כל הערכים בטבלאה
    table = []

    # הוספת הנתונים
    for (k, p), result in results.items():
        table.append([k, p, result['avg_train_error'], result['avg_test_error'], result['error_diff']])

    # הדפסת הטבלה
    headers = ["k", "p", "Average Training Error", "Average Test Error", "Difference"]
    print(tabulate(table, headers=headers, floatfmt=".4f", tablefmt="grid"))


# Main function to run the program
def main():
    # Load data from the provided module
    data, target = Utils.load_data()

    # Define k values and p values to evaluate
    k_values = [1, 3, 5, 7, 9]
    p_values = [1, 2, np.inf]

    # Evaluate k-NN for different k and p values
    results = evaluate_knn(data, target, k_values, p_values)

    # Print the results
    print_results(results)


if __name__ == "__main__":
    main()
