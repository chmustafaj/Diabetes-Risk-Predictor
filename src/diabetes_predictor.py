import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import os

# Load dataset
csvData = []
with open('data/data.csv', 'r') as file:
    csvreader = csv.reader(file)
    next(csvreader)  # Skip header
    for row in csvreader:
        csvData.append(row)

# Convert to NumPy array
dataset = np.array(csvData).astype(float)
np.random.shuffle(dataset)

# Split into 10 folds
folds = np.array_split(dataset, 10)
accuracies = []
threshold = 0.5
lr = 0.1

# Use last fold as test set
test_data = folds[-1]

for j in range(9):
    weights = np.zeros(dataset.shape[1])
    predictions = []
    epoch = 0

    while epoch < 5:
        correct_counter = 0
        for sample in folds[j]:
            a = np.dot(sample[1:], weights[1:])
            prediction = 1 if a > threshold else 0
            correct_counter += (prediction == sample[0])
            weights += lr * (sample[0] - prediction) * np.insert(sample[1:], 0, 1)
        epoch += 1

    # Evaluate on test set
    correct_counter = 0
    predictions = []
    for sample in test_data:
        a = np.dot(sample[1:], weights[1:])
        prediction = 1 if a > threshold else 0
        predictions.append(prediction)
        correct_counter += (prediction == sample[0])

    accuracy = correct_counter / len(test_data)
    accuracies.append(accuracy)

# Final output
print(f"Per-fold Accuracies: {accuracies}")
print(f"Mean Accuracy: {np.mean(accuracies):.2%}")
print("Final Weights:", weights)

# Confusion matrix
actual_values = test_data[:, 0]
conf_matrix = confusion_matrix(actual_values, predictions)

# Plot and save confusion matrix
os.makedirs("results", exist_ok=True)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1])
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("results/confusion_matrix.png")
plt.show()
