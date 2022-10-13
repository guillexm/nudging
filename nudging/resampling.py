import numpy as np

def residual_resampling(weights):
    """
    :arg weights : a numpy array of normalised weights, size N

    returns
    :arg s: an array of integers, size N. X_i will be replaced
    with X_{s_i}.
    """
    # resample Algorithm 3.27
    copies = np.array(np.floor(weights*N), dtype=int)  # x_i = integer fun of len(ensemble)*weight
    N = weights.size
    L = N - np.sum(copies)
    residual_weights = N*weights - copies
    residual_weights /= np.sum(residual_weights)
    
    # Need to add parent indexing 
    for i in range(L):
        u =  np.random.rand()
        cs = np.cumsum(residual_weights) # cumulative sum
        istar = -1
        while cs[istar+1] < u:
            istar += 1
        copies[istar] += 1

    count = 0
    s = np.array(zeros, dtype=int)
    for i in range(N):
        for j in range(copies[i]):
            s[count] = i
            count += 1     
    return s
