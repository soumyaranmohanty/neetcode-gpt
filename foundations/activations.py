import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        arr = np.array(-z)
        value = 1/(1+np.exp(arr))
        #print([float(v) for v in value])
        return np.round([float(v) for v in value], 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        arr=[]
        for a in z:
            arr.append(float(max(0,a)))
        return np.array(arr)
