import numpy as np

def generate(seed, N, r, gamma, dt):
    """
    Parameters
    ----------
    seed : int
        seed for the normal distribution.
    N : int
        total length (discrete time) of the signal.
    r : float
        rate of switching.
    gamma : float
        state: high and low. ie., gamma and -gamma.
    dt : flot
        small time step for the input signal.

    Returns: X,y
    -------
    X: 'numpy.ndarray' of shape (1,N) # input signal 
        -- looks like random noise centered around zero
    y: 'numpy.ndarray' of shape (1,N) # corresponding output signal 
        -- probability of the state; varies in between 0 and 1
    """
    rng = np.random.RandomState(seed)
    xlist = np.ones(N)*gamma
    for i in range(1,N):
        if rng.rand() < r*dt:
            xlist[i] = -xlist[i-1] # switching
        else:
            xlist[i] = +xlist[i-1]
    
    min_L = 1e-8
    max_L = 1-min_L

    sqrt_dt = np.sqrt(dt)
    gamma_2 = np.power(gamma,2)

    dWlist = rng.normal(loc=0.0, scale=1.0, size=N) * sqrt_dt
    dmlist = xlist*dt + dWlist  # INPUT
    L0 = 0.5  # initial probability.
    L = L0 
    dLlist = np.empty(N)
    Llist = np.empty(N)
    for i in range(N):
        dm = dmlist[i]
        dL = 2*gamma*L*(1-L)*dm +2*gamma_2*L*(1-L)*(1-2*L)*dt +r*(1-2*L)*dt # OUTPUT (Rahul)
        L += dL
        applylimit = True
        if applylimit:
            if L<min_L:
                L = min_L
            if L>max_L:
                L = max_L
        dLlist[i] = dL
        Llist[i] = L

    X = column_matrix(dmlist, 1)
    y = Llist.copy()
    #column_matrix(Llist, 1)
    return(X.T, y)

def column_matrix(data, n_column):
    """ to check whether the data is a matrix with n_columns. If not, make it a column matrix of suitable dimension.
    
    Arg:
        data: list, array, or matrix
        n_column: the required number of columns
    
    Return:
        If n_column is equal to 1 AND data is either a list or a one dimensional array ==>> return data as a column vector
        If data is neither a list nor an array ==>> raise error
        If data is a matrix but its number of columns is not same as n_columns ==>> raise error
    """
    if (isinstance(data, list) and n_column ==1): # if data is a list >> make it a column vector
        data = np.array(data)[:,None]
    elif not isinstance(data, np.ndarray): # if not an array >> raise an error
        raise ValueError("Invalid arguement. Data is neither a 'list' nor an 'array'.")
    elif (isinstance(data, np.ndarray) and (data.ndim == 1 == n_column)): # data is an array of dimension 1(= n_column). Good. 
        data = data[:,None] # make it a column vector. 
    elif ((data.ndim > 1) and (data.shape[1] == n_column)): # data is an array of dimension > 1 AND dimension matches.
        pass # This is perfect. do nothing
    else:
        raise ValueError("Invalid arguement. Dimension of data not matching with n_inputs")
    return data

