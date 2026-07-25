import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        m = max(z)
        # print(m)
        arr = np.array([])
        sums=0
        for a in z:
            e=np.exp(a-m)
            # print(e)
            sums +=e
            arr = np.append(arr, float(e))


        # print(arr)
        # print(sums)
        return np.round(arr/sums,4)
