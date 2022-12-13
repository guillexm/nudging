import pytest
from pyop2.mpi import MPI



from nudging.parallel_arrays import in_range

from nudging.parallel_arrays import DistributedDataLayout1D, SharedArray, OwnedArray

#partitions = [4, (2, 3, 4, 2)]


def test_owned_array():
    size = 8
    comm = MPI.COMM_WORLD
    owner = 1

    array = OwnedArray(size, dtype=int, comm=comm, owner=owner)

    assert array.size == size
    assert array.owner == owner
    assert array.comm == comm
    assert array.rank == comm.rank

    if comm.rank == owner:
        assert array.is_owner()
    else:
        assert not array.is_owner()

    # initialise data
    for i in range(size):
        if array.is_owner():
            array[i] = 2*(i+1)
        else:
            assert array[i] == 0

    array.synchronise()
    arr_own = array.data()
    print("arr_own", arr_own)
    # check data
    for i in range(size):
        assert array[i] == 2*(i+1)

    # only owner can modify
    if not array.is_owner():
        with pytest.raises(IndexError):
            array[0] = 0

    # resize
    new_size = 2*size
    array.resize(new_size)

    assert array.size == new_size

    # check original data is unmodified
    for i in range(size):
        assert array[i] == 2*(i+1)

    array.synchronise()
    arr_nown = array.data()
    print("arr_nown", arr_nown)
    # initialise new data
    for i in range(new_size):
        if array.is_owner():
            array[i] = 10*(i-5)

    array.synchronise()

    # check new data
    for i in range(new_size):
        assert array[i] == 10*(i-5)

    array.synchronise()
    arr_neown = array.data()
    print("arr_neown", arr_neown)

test_owned_array()