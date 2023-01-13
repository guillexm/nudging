from pyop2.mpi import MPI
import numpy as np
from nudging.parallel_arrays import DistributedDataLayout1D

partition =  (3, 3, 3, 3)

def target_rank_method(partition):
    comm = MPI.COMM_WORLD
    rank = comm.rank
    print('================== Rank ==================', rank)

    s_arr = np.array([8, 4, 2, 1, 3, 2, 0, 5, 9, 3, 2,10])
    #offset = np.array([0, 2, 9, 11])
    # offset_list
    offset = []
    for i_rank in range(len(partition)):
        offset.append(sum(partition[:i_rank]))
    #print(offset)
    
    layout = DistributedDataLayout1D(partition=partition,  comm=comm)

    for iloc in range(partition[rank]):
        print('ilocal', iloc)
        iglob = layout.transform_index(iloc, itype='l', rtype='g')
        print('iglobal', iglob)
        targt = []
        for j in range(len(s_arr)):
            if s_arr[j] == iglob:
                print('J_val', j)
                for target_rank in range(len(offset)):
                    if offset[target_rank] - j > 0:
                        target_rank -= 1
                        break
                targt.append((j, target_rank))
                print('Target', targt)
                    
target_rank_method(partition)