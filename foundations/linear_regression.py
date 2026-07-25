import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places

        # print(X)
        # print(weights)
        # for i in range(0, len(X)):
        #     pred=0
        #     for j in range(0, len(X[i])):
        #         pred+=X[i][j]*weights

        pred = X @ weights
        # print(pred)
        return np.round(pred, 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places

        loss = model_prediction-ground_truth
        # print(np.mean(loss**2))
        return np.round(np.mean(loss**2), 5)
