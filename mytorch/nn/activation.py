import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        # Compute the softmax in a numerically stable way
        self.axis = self.dim if self.dim >= 0 else self.dim + Z.ndim
        Z_new = Z - np.max(Z, axis=self.axis, keepdims=True)
        exp_Z = np.exp(Z_new)
        sum_Z = np.sum(exp_Z, axis=self.axis, keepdims=True)
        # Apply it to the dimension specified by the `dim` parameter
        self.A = exp_Z / sum_Z
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        
        # Get the shape of the input
        shape = self.A.shape
        dim = self.dim if self.dim >= 0 else self.dim + len(shape)
        # Find the dimension along which softmax was applied
        C = shape[dim]
        self.A = np.moveaxis(self.A, dim, -1)
        dLdA = np.moveaxis(dLdA, dim, -1)
        shape_shift = self.A.shape
        # Reshape input to 2D
        if len(shape) > 2:
            self.A = self.A.reshape(-1, C)
            dLdA = dLdA.reshape(-1, C)
        N_flat, C_flat = dLdA.shape
        dLdZ = np.zeros((N_flat, C_flat))
        for i in range(N_flat):
            J = np.zeros((C_flat, C_flat))
            for m in range(C_flat):
                for n in range(C_flat):
                    J[m, n] = self.A[i][m] * (1 - self.A[i][m]) if m == n else -self.A[i][m] * self.A[i][n]
            dLdZ[i, :] = np.dot(J, dLdA[i])
        # Reshape back to original dimensions if necessary
        if len(shape) > 2:
            # Restore shapes to original
            self.A = self.A.reshape(shape_shift)
            dLdZ = dLdZ.reshape(shape_shift)
        self.A = np.moveaxis(self.A, -1, dim)
        dLdZ = np.moveaxis(dLdZ, -1, dim)
        return dLdZ
 

    