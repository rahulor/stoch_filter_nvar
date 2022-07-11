"""
Created on Sat Oct 23 16:18:56 2021

@author: rahulor@live.com
"""
import numpy as np
    
class NonlinVAR():
    def __init__(self, d, L2penalty=2.5e-6, nonlinearity=1):
        self.d              = d             # input dimension
        self.L2penalty    = L2penalty       # parameter for regularization
        self.nonlinearity = nonlinearity    # 1/2/3 for linear/square/cubic
   
    def configure(self, k, dt, traintime, testtime):
        """
        Parameters
        ----------
        k                   #[int] number of time delay taps
        dlin = k*d          #[int] size of linear part of feature vector
        dnonlin = int(dlin*(dlin+1)/2)  #[int] size of nonlinear part of feature vector
        dtot = 1 + dlin + dnonlin #[int] total size of feature vector: constant + lin + nonlin
        dt                          # [s] time step of the signal (same dt used in data.py)
        warmup = int(2.0 + k*dt)            # [s] time to warm up NVAR.  warmup > k*dt
        traintime                           # [s] time to train for
        testtime                            # [s] time to test for
        maxtime = warmup+traintime+testtime # [s] total time to run for
        
        Returns
        -------
        None.
        """
        self.k          = k
        self.dlin       = k*self.d  
        self.d2         = int(self.dlin*(self.dlin+1)/2)
        self.d3         = int( self.dlin*(self.dlin+1)*(self.dlin+2)/6 )
        #
        warmup          = int(1.0 + k*dt)
        # discrete-time versions of the times defined above            
        self.warmup_pts = round(warmup/dt)
        self.traintime_pts  = round(traintime/dt)
        self.warmtrain_pts  = self.warmup_pts + self.traintime_pts
        self.testtime_pts   = round(testtime/dt)
        self.warmtest_pts   = self.warmup_pts + self.testtime_pts

    def fit(self, X_train, y_train):
        dtot = X_train.shape[0]
        y_train = y_train.reshape(1,-1)
        beta = self.L2penalty 
        
        W_out = y_train @ X_train.T @ np.linalg.pinv(X_train @ X_train.T + beta*np.identity(dtot))
        y_train_pred = W_out @ X_train
        self.y_train_pred = y_train_pred.reshape(-1)
        self.W_out = W_out
        return()
        
    def train(self, X, y):
        """
        * construct the feature matrix O_train;
        Obtain W_out from y_tar and O_train : ( y_tar = W_out @ O_train )
        Compute y_train_pred
        
        Parameters
        ----------
        X : 'numpy.ndarray' of shape (d, warmtrain_pts) # input signal
        y : 'numpy.ndarray' of shape (warmtrain_pts, ) # output signal

        Returns
        -------
        None.

        """
        O_train = self.feature_vector(X)
        y_tar  = y[self.warmup_pts:]
        self.fit(O_train, y_tar)
        return()
    def test(self, X):
        """
        * construct the feature matrix O_test;
        W_out is already found in train()
        Compute y_test_pred
        
        Parameters
        ----------
        X : 'numpy.ndarray' of shape (d, warmtrain_pts) # input signal

        Returns
        -------
        None.

        """
        O_test = self.feature_vector(X)
        self.y_test_pred  = (self.W_out @ O_test).reshape(-1)
        return()
    
    def feature_vector(self, X):
        """
        * construct linear delaytaps 
        * take squares (without repetition) from linear delaytaps if self.nonlinear = 2
        * take cubes   (without repetition) from linear delaytaps if self.nonlinear = 3
        vstack everything to O_total and return it.
        Parameters
        ----------
        X : 'numpy.ndarray' of shape (d, warmtrain_pts or warmtest_pts) # input signal
            DESCRIPTION.

        Returns
        -------
        O_total

        """
        start = self.warmup_pts
        stop  = X.shape[1]
        O_const  = np.ones((1, stop-start))
        O_linear = self.linear_delaytaps(X)
        if self.nonlinearity==1:
            O_total = np.vstack((O_const, O_linear))
        elif self.nonlinearity==2:
            O_quadr = self.polynomial_2nd_order(O_linear)
            O_total  = np.vstack((O_const, O_linear, O_quadr))
        elif self.nonlinearity==3:
            O_quadr = self.polynomial_2nd_order(O_linear)
            O_cubic = self.polynomial_3rd_order(O_linear)
            O_total  = np.vstack((O_const, O_linear, O_quadr, O_cubic))
        else:
            pass
        return(O_total)
    def linear_delaytaps(self, M):
        """
        complete delay taps are constructed; from warmup_pts to M.shape[1].
        linear delay taps of M with k delay terms looks like a sandwitch of
        - M[:, start:stop]
        - M[:, start-1:stop-1]
        - M[:, start-2:stop-2]
        :
        upto
        - M[:, start-(k-1):stop-(k-1)]
        
        Parameters
        ----------
        M : 'numpy.ndarray' of shape (d,N) # input signal
        
        Returns
        -------
        linear delaytaps of M, with k delay terms.
        """
        start = self.warmup_pts
        stop  = M.shape[1]
        O_linear = np.empty((self.dlin, stop-start))
        for delay in range(self.k):
            O_linear[delay*self.d:(delay+1)*self.d,:] = M[:, start-delay:stop-delay]
        return(O_linear)
    def polynomial_2nd_order(self, M):
        """
        M receives O_linear. Need to construct all the products of rows without repetition.
        step 1: i=0
        step 2: let A = M[i] (i-th row)
        step 3: go through all the rows from M[i] -- say B, and store A*B
        step 4: increment i and go to step 2 utill i covers all the rows in M

        Parameters
        ----------
        M : 'numpy.ndarray' of shape (dlin, traintime_pts or testtime_pts) # O_linear

        Returns
        -------
        O_nonlin: nonlinear terms (products) constructed out of O_linear

        """
        O_nonlin = np.empty((self.d2, M.shape[1]))
        row = 0
        for i, A in enumerate(M):
            for B in M[i:]:
                O_nonlin[row] = A*B
                row += 1
        return(O_nonlin)
    def polynomial_3rd_order(self, M):
        """
        M receives O_linear. Need to construct all monomials of degree 3, without repetition.

        Parameters
        ----------
        M : 'numpy.ndarray' of shape (dlin, traintime_pts or testtime_pts) # O_linear

        Returns
        -------
        O_nonlin: nonlinear terms (cubic monomials) constructed out of O_linear

        """
        O_nonlin = np.empty((self.d3, M.shape[1]))
        index = np.arange(M.shape[0])
        row = 0
        for i in index:
            for j in index[i:]:
                for k in index[j:]:
                    O_nonlin[row] = M[i]*M[j]*M[k]
                    row += 1
        return(O_nonlin)
    
if __name__ == '__main__':
    pass
