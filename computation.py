"""
- create different data set for training and testing
- data.generate returns input signal and probability. Note that 0 < p < 1.
- convert this probability p to logit(p)
- use logit(p) for ridge regression.
- predicted output y is then converted back to probability using sigmoid(y); pretty cool, no ?

We call this 'logit-approach' in the manuscript: 
https://doi.org/10.48550/arXiv.2206.13018
"""
import time
import data
from class_nvar import NonlinVAR
from class_probmapping import probmap
import numpy as np
import pandas as pd
from inputs import r, gamma, dt
from config import d, L2penalty, k, traintime, testtime, nonlinearity
from sklearn.metrics import mean_squared_error

def apply_limit(A):
    min_A = 1e-8
    max_A = 1-min_A
    A[A < min_A] = min_A
    A[A > max_A] = max_A
    return(A)
  
def divergence_KL(L,B):
    DKL = L*np.log(L/B) + (1-L)*np.log((1-L)/(1-B))
    return(DKL)
def accuracy(ytrue, ypred):
    # arguement ytrue and ypred is in (-inf, inf)
    ptrue, ppred = probmap.y_to_prob(ytrue), probmap.y_to_prob(ypred)
    ppred = apply_limit(ppred) # to confirm it is inside (0,1) before finding DKL.
    
    mse = mean_squared_error(ptrue, ppred)
    DKL = np.mean( divergence_KL(ptrue, ppred) )
    
    ser = pd.Series([mse, DKL], index=['mean_squared_error', 'Kullback-Leibler divergence'])
    print(ser.to_string())
    return(ser)
def run():  
    print('running ...')
    nvar = NonlinVAR(d=d, L2penalty=L2penalty, nonlinearity=nonlinearity)
    nvar.configure(k, dt, traintime, testtime)
    warmup_pts, warmtrain_pts = nvar.warmup_pts, nvar.warmtrain_pts
    maxtime_pts = warmup_pts+warmtrain_pts
    #==========================================================================
    #training process
    N = nvar.warmtrain_pts
    seed = 40
    X, p = data.generate(seed, N, r, gamma, dt)
    y = probmap.prob_to_y(p)
    # here we use y for the regression; p = (0,1);  y = (-inf, inf)
    # -------------------------------------------------- training process ...
    start_time = time.time()
    nvar.train(X, y)
    end_time = time.time()
    time_elapsed_train = (end_time - start_time)
    
    print('='*20, 'training error'.center(15,' '), '='*20)
    y_tar = y[warmup_pts:]
    ser_train_error = accuracy(y_tar , nvar.y_train_pred )
    print('time_elapsed (train)'.ljust(30,' '), np.round(time_elapsed_train,3), '[s]')
    # =========================================================================
    N = nvar.warmtest_pts
    seed = seed+1
    X, p = data.generate(seed, N, r, gamma, dt) 
    y = probmap.prob_to_y(p)
    # -------------------------------------------------- testing process ...
    start_time = time.time()
    nvar.test(X)
    end_time = time.time()
    time_elapsed_test = (end_time - start_time)
    
    print('='*20, 'testing error'.center(15,' '), '='*20)
    y_test = y[warmup_pts:]
    ser_test_error = accuracy(y_test, nvar.y_test_pred)
    print('time_elapsed (test)'.ljust(30,' '), np.round(time_elapsed_test,3), '[s]')
    time_elapsed = time_elapsed_train + time_elapsed_test
    #==========================================================================
    # writing data set to file for later use
    warmtest_pts = nvar.warmtest_pts
    maxtime_pts = warmup_pts+warmtest_pts
    tlist = np.linspace(0, maxtime_pts*dt, maxtime_pts, endpoint=False)
    t1, t2 = warmup_pts, warmtest_pts
    # converting back to probability before writing
    p_true, p_pred = probmap.y_to_prob(y_test), probmap.y_to_prob(nvar.y_test_pred)
    testdata = np.vstack((tlist[t1:t2], X[:,t1:], p_true, p_pred))
    df_testdata = pd.DataFrame(testdata.T, columns=['time', 'u', 'true', 'pred'])
    outputpath = 'data/testdata.csv'
    df_testdata.to_csv(outputpath)
    
    # writing weight to file for later use
    Wout = nvar.W_out
    df_weight = pd.DataFrame(Wout.T, columns=['Wout'])
    outputpath = 'data/weight.csv'
    df_weight.to_csv(outputpath)
    
    return(time_elapsed, ser_train_error, ser_test_error)
    
if __name__ == '__main__':
    pass
    
    
    