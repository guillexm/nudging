import pytest
from pyop2.mpi import MPI



from nudging.parallel_arrays import in_range

from nudging.parallel_arrays import DistributedDataLayout1D, SharedArray, OwnedArray

#partitions = [4, (2, 3, 4, 2)]



def test_distributed_data_layout(partition):
    comm = MPI.COMM_WORLD
    layout = DistributedDataLayout1D(partition, comm=comm) # layout is just an object

    if isinstance(partition, int): # if partion is integer
        partition = tuple(partition for _ in range(layout.comm.size))

    rank = layout.rank
    print("rank",rank)
    nlocal = partition[rank]
    nglobal = sum(partition)
    offset = sum(partition[:rank])
    #print("nlocal", nlocal)
    #print("nglobal",nglobal)
    # check attributes
    assert layout.partition == partition
    assert layout.rank == comm.rank
    assert layout.local_size == nlocal
    assert layout.global_size == nglobal
    assert layout.offset == offset

    max_indices = {
        'l': nlocal,
        'g': nglobal
    }

    # shift index: from_range == to_range
    for itype in max_indices.keys():

        imax = max_indices[itype]
        i = imax - 1
        #print(i)
        # +ve index unchanged
        pos_shift = layout.transform_index(i, itype=itype, rtype=itype)
        assert (pos_shift == i)

        # -ve index changed to +ve
        neg_shift = layout.transform_index(-i, itype=itype, rtype=itype)
        #print(neg_shift)
        assert (neg_shift == imax - i)

    # local address -> global address

    ilocal = 1

    # +ve index in local range
    iglobal = layout.transform_index(ilocal, itype='l', rtype='g')
    assert (iglobal == offset + ilocal)

    # -ve index in local range
    iglobal = layout.transform_index(-ilocal, itype='l', rtype='g')
    assert (iglobal == offset + nlocal - ilocal)

    ilocal = nlocal + 1

    # +ve index out of local range
    with pytest.raises(IndexError):
        iglobal = layout.transform_index(ilocal, itype='l', rtype='g')

    # -ve index out of local range
    with pytest.raises(IndexError):
        iglobal = layout.transform_index(-ilocal, itype='l', rtype='g')

    # global address -> local address

    # +ve index in range
    iglobal = offset + 1
    ilocal = layout.transform_index(iglobal, itype='g', rtype='l')
    assert (ilocal == 1)

    assert layout.is_local(iglobal)

    # -ve index in range
    iglobal = -sum(partition) + offset + 1
    ilocal = layout.transform_index(iglobal, itype='g', rtype='l')
    assert (ilocal == 1)

    assert layout.is_local(iglobal)

    # +ve index out of local range
    iglobal = (offset + nlocal + 1) % nglobal
    with pytest.raises(IndexError):
        ilocal = layout.transform_index(iglobal, itype='g', rtype='l')

    assert not layout.is_local(iglobal)

    with pytest.raises(IndexError):
        layout.is_local(iglobal, throws=True)

    # -ve index out of local range
    iglobal = -(nglobal - (offset + nlocal))
    with pytest.raises(IndexError):
        ilocal = layout.transform_index(iglobal, itype='g', rtype='l')

    assert not layout.is_local(iglobal)

    with pytest.raises(IndexError):
        layout.is_local(iglobal, throws=True)



test_distributed_data_layout(3)

