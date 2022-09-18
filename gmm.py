import numpy as np
from tqdm import tqdm
from kmeans import KMeans


SIGMA_CONST = 1e-6 # Only add SIGMA_CONST when sigma_i is not invertible
LOG_CONST = 1e-32

FULL_MATRIX = False # Set False if the covariance matrix is a diagonal matrix

class GMM(object):
    def __init__(self, X, K, max_iters=100):  # No need to change
        """
        Args:
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.points = X
        self.max_iters = max_iters

        self.N = self.points.shape[0]  # number of observations
        self.D = self.points.shape[1]  # number of features
        self.K = K  # number of components/clusters

    # Helper function for you to implement
    def softmax(self, logit):  # [5pts]
        N = logit.shape[0]
        D = logit.shape[1]
        maxlogit = (np.ones((D,N)) * np.max(logit, axis = 1)).T
        logit = logit - maxlogit
        explogit = np.exp(logit)
        sumexp = explogit.sum(axis = 1)
        prob = (explogit.T * (sumexp**-1)).T
        return prob
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array. See the above function.
        """




    def logsumexp(self, logit):  # [5pts]

        maximums = np.max(logit,axis=1)
        logit_scaled = logit - maximums[:,None]
        exponential = np.exp(logit_scaled)
        summation = np.sum(exponential,axis=1)
        log = np.log(summation)
        s = (log+maximums).reshape((logit.shape[0], 1))

        return s



        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
        """




    # for undergraduate student
    def normalPDF(self, logit, mu_i, sigma_i):  # [5pts]

        sigma = np.diagonal(sigma_i)
        step1 = np.exp(np.square(logit - mu_i)/(-2*(sigma)))
        step2 = 1/np.sqrt(2*np.pi*(sigma))

        return np.prod(step1*step2,axis =1)



        """
        Args:
            logit: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            np.diagonal() should be handy.
        """


    def _init_components(self, **kwargs):  # [5pts]
        K = self.K
        N = self.points.shape[0]
        D = self.points.shape[1]
        pi = np.ones(K) * (K ** -1)
        mu = KMeans()._init_centers(self.points, K)
        sigma = np.zeros((K, D, D))
        for k in range(K):
            sigma[k] = np.identity(D)
        return pi, mu, sigma
        """
        Args:
            kwargs: any other arguments you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
                You will have KxDxD numpy array for full covariance matrix case
        """

    def _ll_joint(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):  # [10 pts]



        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.

        Return:
            ll(log-likelihood): NxK array, where ll(i, k) = log pi(k) + log NormalPDF(points_i | mu[k], sigma[k])
        """
        # === graduate implementation
        #if full_matrix is True:
            #...

        # === undergraduate implementation
        if full_matrix is False:
            K = mu.shape[0]
            x = self.points
            D = x.shape[1]
            N = x.shape[0]

            ll = np.ndarray((K,N))

            for k in range(K):
                logpi = np.log(pi[k] + 1e-32)
                PDF = np.log(self.normalPDF(x,mu[k],sigma[k]) + 1e-32) + logpi
                ll[k] = PDF
            ll = np.transpose(ll)
            return ll



    def _E_step(self, pi, mu, sigma,  full_matrix=FULL_MATRIX, **kwargs):  # [5pts]

        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.

        Hint:
            You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above.
        """
        # === graduate implementation
        #if full_matrix is True:


        # === undergraduate implementation
        if full_matrix is False:
            ll = self._ll_joint(pi, mu, sigma)
            gamma = self.softmax(ll)
            return gamma



    def _M_step(self, gamma, full_matrix=FULL_MATRIX, **kwargs):  # [10pts]

        N = self.points.shape[0]
        D = self.points.shape[1]
        K = gamma.shape[1]

        Nk = gamma.sum(axis = 0)
        mu = ((gamma.T).dot(self.points).T / Nk).T
        pi = Nk / (gamma.sum())

        sigma = np.ones((K, D, D))
        for k in range(K):
            mumatrix = np.ones((N, D)) * mu[k, :]
            xminmu = self.points - mumatrix
            sigmamatrix = (((gamma[:, k] * (xminmu).T)).dot(xminmu)) / gamma[:, k].sum()
            sigma[k, :, :] = np.identity(D) * sigmamatrix
        return(pi, mu, sigma)


        """
        Args:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case

        Hint:
            There are formulas in the slides and in the Jupyter Notebook.
        """

        # === graduate implementation
        #if full_matrix is True:
            # ...

        # === undergraduate implementation
        #if full_matrix is False:
            # ...


    def __call__(self, full_matrix=FULL_MATRIX, abs_tol=1e-16, rel_tol=1e-16, **kwargs):  # No need to change
        """
        Args:
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want

        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)

        Hint:
            You do not need to change it. For each iteration, we process E and M steps, then update the paramters.
        """
        pi, mu, sigma = self._init_components(**kwargs)
        pbar = tqdm(range(self.max_iters))

        for it in pbar:
            # E-step
            gamma = self._E_step(pi, mu, sigma, full_matrix)

            # M-step
            pi, mu, sigma = self._M_step(gamma, full_matrix)

            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(pi, mu, sigma, full_matrix)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description('iter %d, loss: %.4f' % (it, loss))
        return gamma, (pi, mu, sigma)
