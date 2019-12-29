import numpy as np

class admm_nmr():
    """ admm algorithm for NMR
    
    fitting the optimal regression coefficient for the image dataset. 
    
    Parameters
    ----------
    A : array, shape=[n_samples, horizontal pixels, vertical pixels]
    
    B : array, shape=[horizontal pixels, vertical pixels]
    
    x : array, shape=[n_samples]
        Representation of regression coefficient vector for each image.
    
    lambd : float, deault=1.0
    mu : float, deault=1.0
    gamma : float, deault=1.0
    ep_abs : float, deault=1.0
    ep_rel : float, deault=1.0 
    max_iter : float, deault=10
    
     Attributes
    ----------
    x_new : array, shape=[n_samples]
    Representation of the optimal regression coefficient vector for each image.
    
     Examples
    --------
    >>> from nmr_classifier.admm_nmr import admm_nmr
    >>> 
    
    """
    def __init__(self, lambd=1.0, mu=1.0, ep_abs=1.0, ep_rel=1.0, max_iter=10):
        self.lambd = lambd
        self.mu = mu
        self.ep_abs = ep_abs
        self.ep_rel = ep_rel
        self.max_iter = max_iter

    
    def coef_img(self,A,x):
        A_x = np.zeros((A.shape[1], A.shape[2]))
        n = A.shape[0]
        for num in range(n):
            A_x += x[num]*A[num]
        return A_x
    
    def vec(self,input_array):
        return input_array.T.reshape(-1,1).T.tolist()[0]
    
    def svt(self, A_x, B, Z):
        svt_in = A_x - B + 1/self.mu*Z
        U, S, V = np.linalg.svd(svt_in, full_matrices=False)
        S = np.maximum(S - 1/self.mu, 0)
        Y = np.linalg.multi_dot([U, np.diag(S), V])
        return Y
    
    def fit(self, A, B):
        print('Start update')
        print('------------------------------------------------------------------------')
        n = A.shape[0]
        p = A.shape[1]
        q = A.shape[2]
        
        # Step 1
        H_T = np.zeros((p*q, n)).T
        for n_iter in range(n) :
            H_T[n_iter] = self.vec(A[n_iter])
        H = H_T.T
        M = np.dot(np.linalg.inv(np.dot(H.T,H)+self.lambd/self.mu*np.identity(H.shape[1])), H.T)
        
        
        # Step 2
        Y = -B
        Z = 0
        k = 0
        x = np.zeros(n)
        
        
        # Step 3 ~ 5
        for iter_loop in range(self.max_iter):

            # Updating x
            g_in = B + Y - 1/self.mu*Z
            g = self.vec(g_in)
            x_new = np.dot(M, g)

            # Updating Y
            A_x_new = self.coef_img(A,x_new)
            Y_new = self.svt(A_x_new, B, Z)

            # Updating Z
            Z_new = Z + self.mu*(A_x_new - Y_new - B)

            # termination 
            r_pri = A_x_new - Y_new - B
            ep_pri = np.sqrt(p*q)*self.ep_abs + self.ep_rel*max(np.linalg.norm(A_x_new,ord='nuc'), np.linalg.norm(Y_new,ord='nuc'), np.linalg.norm(B,ord='nuc'))
            s_dual = 1/self.mu * np.dot(H.T, self.vec(Y-Y_new))
            ep_dual = np.sqrt(n)*self.ep_abs + self.ep_rel*np.linalg.norm(np.dot(H.T, Z_new.T.reshape(-1,1).T.tolist()[0]),ord=2)


            if(np.linalg.norm(r_pri, ord=2)<=ep_pri and np.linalg.norm(s_dual, ord=2)<=ep_dual):
                    break    # break here


            x = x_new
            Y = Y_new
            Z = Z_new
            A_x = A_x_new

            print(str(iter_loop+1) + ' update ')
            print(str(x))
            print('')

        print('------------------------------------------------------------------------')
        print('End update')
        
        return x_new