#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

def pageRank(M, num_iter: int = 100, d: float = 0.85):
    """
    Parameters
    ----------
    M : numpy array
        adjacency matrix where M[i,j] represents the link from 'j' to 'i', such that for all 'j' -> sum(i, M[i,j]) = 1
    num_iter : int, optional
        number of iterations (default 100)
    d : float, optional
        damping factor (default 0.85)

    Returns
    -------
    numpy array
        vector of ranks such that v[i] is the i-th rank from [0, 1],
        v sums to 1
    """
    N = M.shape[1]
    v = np.ones(N) / N
    M_hat = (d * M + (1 - d) / N)
    for i in range(num_iter):
        v = M_hat @ v
    return v

M = np.array([[0, 0, 0, 0, 1],
              [0.5, 0, 0, 0, 0],
              [0.5, 1, 0, 0, 0],
              [0, 0, 1, 0.5, 0],
              [0, 0, 0, 0.5, 0]])
v = pageRank(M, 100, 0.85)
print(v)


# In[ ]:




