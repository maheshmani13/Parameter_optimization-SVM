from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np


def svm_optimization(sample_data, iterations):
    X = sample_data.drop(columns=['Class'])
    y = sample_data['Class']

    Cm_iterations=iterations
    best_accuracy=0
    best_params={}
    accuracy_values=[]

    for j in range(0, Cm_iterations):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        # Randomly select kernel and parameters
        kernel = np.random.choice(['linear', 'rbf', 'sigmoid'])
        C = np.random.uniform(0, 10)
        gamma = np.random.uniform(0, 10)
        print(kernel)
        # Train the SVM model
        model = SVC(kernel=kernel, gamma=gamma, C=C)
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)


        # Update best accuracy and corresponding parameters if current model is better
        if accuracy > best_accuracy:
            best_accuracy = accuracy

            best_params = {'kernel': kernel, 'gamma': gamma, 'C': C}
        accuracy_values.append(best_accuracy)

    return [best_params, best_accuracy, accuracy_values]
