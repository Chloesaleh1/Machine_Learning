import numpy as np

class ImgCompression(object):
    def __init__(self):
        pass

    def svd(self, X): # [5pts]
        """
        Do SVD. You could use numpy SVD.
        Your function should be able to handle black and white
        images ((N,D) arrays) as well as color images ((N,D,3) arrays)
        In the image compression, we assume that each column of the image is a feature. Perform SVD on the channels of
        each image (1 channel for black and white and 3 channels for RGB)
        Image is the matrix X.

        Args:
            X: (N,D) numpy array corresponding to black and white images / (N,D,3) numpy array for color images

        Return:
            U: (N,N) numpy array for black and white images / (N,N,3) numpy array for color images
            S: (min(N,D), ) numpy array for black and white images / (min(N,D),3) numpy array for color images
            V^T: (D,D) numpy array for black and white images / (D,D,3) numpy array for color images
        """

        N = X.shape[0]
        D = X.shape[1]

        if X.ndim == 2:
            U, S, V = np.linalg.svd(X)
        else:
            U = np.ones((N, N, 3))
            S = np.ones((min(N, D), 3))
            V = np.ones((D, D, 3))
            for i in range(3):
                U[:, :, i], S[:, i], V[:, :, i] = np.linalg.svd(X[:, :, i])
        return U, S, V


    def rebuild_svd(self, U, S, V, k): # [5pts]
        """
        Rebuild SVD by k componments.

        Args:
            U: (N,N) numpy array for black and white images / (N,N,3) numpy array for color images
            S: (min(N,D), ) numpy array for black and white images / (min(N,D),3) numpy array for color images
            V: (D,D) numpy array for black and white images / (D,D,3) numpy array for color images
            k: int corresponding to number of components

        Return:
            Xrebuild: (N,D) numpy array of reconstructed image / (N,D,3) numpy array for color images

        Hint: numpy.matmul may be helpful for reconstructing color images
        """
        N = U.shape[0]
        D = V.shape[0]

        if U.ndim == 2:
            U = U[:, 0:k]
            S = S[0:k] * np.identity(k)
            V = V[0:k, :]
            Xnew = np.matmul(np.matmul(U, S), V)
        elif U.ndim == 3:
            Xnew = np.ones((N, D, 3))
            for i in range(3):
                A = U[:, 0:k, i].reshape((N, k))
                B = S[0:k, i].reshape(k) * np.identity(k)
                C = V[0:k, :, i].reshape((k, D))
                Xnew[:, :, i] = np.matmul(np.matmul(A, B), C)
        return Xnew


    def compression_ratio(self, X, k): # [5pts]
        """
        Compute the compression ratio of an image: (num stored values in compressed)/(num stored values in original)

        Args:
            X: (N,D) numpy array corresponding to black and white images / (N,D,3) numpy array for color images
            k: int corresponding to number of components

        Return:
            compression_ratio: float of proportion of storage used by compressed image
        """
        first = X.shape[1] * X.shape[0]
        compressed = k * (1 + X.shape[0] + X.shape[1])
        return compressed / first



    def recovered_variance_proportion(self, S, k): # [5pts]
        """
        Compute the proportion of the variance in the original matrix recovered by a rank-k approximation

        Args:
           S: (min(N,D), ) numpy array black and white images / (min(N,D),3) numpy array for color images
           k: int, rank of approximation

        Return:
           recovered_var: float (array of 3 floats for color image) corresponding to proportion of recovered variance
        """
        S = S ** 2

        if S.ndim == 1:
            recovered_var = S[0:k].sum() / S.sum()
        else:
            recovered_var = S[0:k, :].sum(axis = 0) / S.sum(axis = 0)
        return recovered_var

