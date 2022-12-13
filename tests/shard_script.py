import pytest
from pyop2.mpi import MPI



from nudging.parallel_arrays import in_range

from nudging.parallel_arrays import DistributedDataLayout1D, SharedArray, OwnedArray

#partitions = [4, (2, 3, 4, 2)]


def test_shared_array(partition):
    comm = MPI.COMM_WORLD
    rank = comm.rank
    #print(rank)

    if isinstance(partition, int):
        partition = tuple(partition for _ in range(comm.size))
    array = SharedArray(partition=partition, dtype=int, comm=comm)
    #print(partition)
    if isinstance(partition, int):
        partition = tuple(partition for _ in range(comm.size))

    local_size = partition[rank]
    #print(local_size)
    offset = sum(partition[:rank])
    #print("offset", offset)

    # check everything is zero to start
    for i in range(array.global_size):
        assert array.dglobal[i] == 0 # due to _data def

    # each rank sets its own elements using local access
    for i in range(local_size):
        array.dlocal[i] = array.rank + 1

    # synchronise
    array.synchronise()
    arr_sc = array.data()
    print(arr_sc)

    # check all elements are correct
    for rank in range(array.comm.size):
        check = rank + 1
        offset = sum(partition[:rank])
        for i in range(partition[rank]):
            j = offset + i
            assert array.dglobal[j] == check

    # each rank sets its own elements using global access
    for i in range(local_size):
        j = array.offset + i
        array.dglobal[j] = (array.rank + 1)*2

    array.synchronise()
    arr_gsc = array.data()
    print(arr_gsc)
    # check all elements are correct
    for i in range(local_size):
        assert array.dlocal[i] == (array.rank + 1)*2

    for rank in range(array.comm.size):
        check = (rank + 1)*2
        offset = sum(partition[:rank])
        for i in range(partition[rank]):
            j = offset + i
            assert array.dglobal[j] == check

test_shared_array(5)

#test_shared_array((2, 3, 4, 2))