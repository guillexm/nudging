import pytest
from pyop2.mpi import MPI

import numpy as np

from nudging.parallel_arrays import in_range

from nudging.parallel_arrays import DistributedDataLayout1D, SharedArray, OwnedArray

nensemble = [4, (2, 3, 4, 2)]




def assimilation_step(nensemble):

    comm = MPI.COMM_WORLD
    rank = comm.rank
    

    if isinstance(nensemble, int):
        nensemble = tuple(nensemble for _ in range(comm.size))

    loc_size = nensemble[rank]
    #print(loc_size)
    N = loc_size
    global_size = int(np.sum(nensemble))

    #Shared array
    weight_arr = SharedArray(partition=nensemble, dtype=int, comm=comm)
        
    # forward model step
    for i in range(N):
        weight_arr.dlocal[i] = (i+1)

    # Synchronising weights to rank 0
    weight_arr.synchronise(root=0)

    weights = weight_arr.data()
    print("Rank", rank)
    print("weights", weights)

    #s = np.zeros(global_size)
    s_arr = OwnedArray(size = global_size, dtype=int, comm=comm, owner=0)
    
    if rank == 0:
        for i in range(global_size):
            s_arr[i]=weights[i]**2
    
    s_arr.synchronise()
    s = s_arr.data()

    print("s", s)

assimilation_step(6)