from abc import ABCMeta, abstractmethod, abstractproperty
from firedrake import *
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from .resampling import *
import numpy as np
from scipy.special import logsumexp

from .parallel_arrays import DistributedDataLayout1D, SharedArray, OwnedArray


class base_filter(object, metaclass=ABCMeta):
    ensemble = []
    new_ensemble = []

    def __init__(self):
        pass

    def setup(self, nensemble, model):
        """
        Construct the ensemble

        nensemble - a list of the number of ensembles on each ensemble rank
        model - the model to use
        """
        self.model = model
        self.nensemble = nensemble
        n_ensemble_partitions = len(nensemble)
        self.nspace = int(COMM_WORLD.size/n_ensemble_partitions)
        assert(self.nspace*n_ensemble_partitions == COMM_WORLD.size)

        self.subcommunicators = Ensemble(COMM_WORLD, self.nspace)
        # model needs to build the mesh in setup
        self.model.setup(self.subcommunicators.comm)
        if isinstance(nensemble, int):
            nensemble = tuple(nensemble for _ in range(self.subcommunicators.comm.size))
        
        # setting up ensemble 
        self.ensemble_rank = self.subcommunicators.ensemble_comm.rank
        self.ensemble_size = self.subcommunicators.ensemble_comm.size
        self.ensemble = []
        self.new_ensemble = []
        for i in range(self.nensemble[self.ensemble_rank]):
            self.ensemble.append(model.allocate())
            self.new_ensemble.append(model.allocate())

        # some numbers for shared array and owned array
        self.nlocal = self.nensemble[self.ensemble_rank]
        self.nglobal = int(np.sum(self.nensemble))
            
        # Shared array for the weights
        self.weight_arr = SharedArray(partition=self.nensemble, dtype=float,
                                      comm=self.subcommunicators.ensemble_comm)
        # Owned array for the resampling protocol
        self.s_arr = OwnedArray(size = self.nglobal, dtype=int,
                                comm=self.subcommunicators.ensemble_comm,
                                owner=0)
        # data layout for coordinating resampling communication
        self.layout = DistributedDataLayout1D(self.nensemble,
                                         comm=self.subcommunicators.ensemble_comm)

        # offset_list
        self.offset_list = []
        for i_rank in range(len(self.nensemble)):
            self.offset_list.append(sum(self.nensemble[:i_rank]))

        

                
        #a resampling method
        self.resampler = residual_resampling
        
    def parallel_resample(self):
        
        
        # Synchronising weights to rank 0
        self.weight_arr.synchronise(root=0)
        if self.ensemble_rank == 0:
            weights = self.weight_arr.data()
            # renormalise
            weights = np.exp(-weights)
            weights /= np.sum(weights)
            self.ess = 1/np.sum(weights**2)

        # compute resampling protocol on rank 0
        if self.ensemble_rank == 0:
            s = self.resampler(weights, model=SimModel)
            for i in range(self.nglobal):
                self.s_arr[i]=s[i]

        # broadcast protocol to every rank
        self.s_arr.synchronise()
        s_copy = self.s_arr.data()
        print('=========================================Rank====================================', self.ensemble_rank)
        print('s', s_copy)

        # Fix send and recv of ensemble Communication stage
        # we need a list of which ensemble members we will receive from
        #         also which local ensemble member they be copied to
        #         the global number of this ensemble member is the tag
        #         also which ensemble member we are receiving from
        mpi_requests = []
        # loop over local ensemble members, doing sends and receives
        for ilocal in range(self.nensemble[self.ensemble_rank]):
            print('ilocal', ilocal)
            # get the global ensemble index
            iglobal = self.layout.transform_index(ilocal, itype='l',
                                             rtype='g')
            print('iglobal', iglobal)
            # work on send list
            # find all j such that s[j] = iglobal
            targets = []
            for j in range(self.s_arr.size):
                if s_copy[j] == iglobal:
                    print('J_val', j)
                    # want to get the ensemble rank of each global index
                    for target_rank in range(len(self.offset_list)):
                        if self.offset_list[target_rank] - j > 0:
                            target_rank -= 1
                            break
                    targets.append((j, target_rank))
                    print('Target', targets)
            for target in targets:
                if target[1] == self.ensemble_rank:
                    jlocal = self.layout.transform_index(target[0],
                                                         itype='g',
                                                         rtype='l')
                    self.new_ensemble[jlocal].assign(self.ensemble[ilocal])
                else:
                    request_send = self.subcommunicators.isend(
                        self.ensemble[ilocal], dest=target[1], tag=target[0])
                    mpi_requests.extend(request_send)


            for source_rank in range(len(self.offset_list)):
                if self.offset_list[source_rank] - s_copy[iglobal] > 0:
                    source_rank -= 1
                    break

            if source_rank != self.ensemble_rank:
                request_recv = self.subcommunicators.irecv(
                    self.new_ensemble[ilocal],
                    source=source_rank,
                    tag=iglobal)
                mpi_requests.extend(request_recv)

        MPI.Request.Waitall(mpi_requests)
        # copy back into ensemble for the next iteration
        for i in range(self.nlocal):
            self.ensemble[i].assign(self.new_ensemble[i])

        
    @abstractmethod
    def assimilation_step(self, y, log_likelihood):
        """
        Advance the ensemble to the next assimilation time
        and apply the filtering algorithm
        y - a k-dimensional numpy array containing the observations
        log_likelihood - a function that computes -log(Pi(y|x))
                         for computing the filter weights
        """
        pass

class sim_filter(base_filter):

    def __init__(self):
        super().__init__()

    def assimilation_step(self, y, log_likelihood):
        N = len(self.ensemble)
       # forward model step
        for i in range(N):
            self.model.run(self.ensemble[i])   
            # particle weights
            Y = self.model.obs()
            self.weight_arr.dlocal[i] = log_likelihood(y-Y)

        # do the resampling and communication
        self.parallel_resample()

class bootstrap_filter(base_filter):
    def __init__(self, nsteps, noise_shape):
        self.nsteps = nsteps
        self.noise_shape = noise_shape

    def assimilation_step(self, y, log_likelihood):
        N = len(self.ensemble)
        
       # forward model step
        for i in range(N):
            W = np.random.randn(*(self.noise_shape))
            self.model.run(self.nsteps, W,
                             self.ensemble[i], self.ensemble[i])   # solving FEM with ensemble as input and final sol ensemble

            # particle weights
            Y = self.model.obs(self.ensemble[i])
            self.weight_arr.dlocal[i] = log_likelihood(y-Y)

        # do the resampling and communication
        self.parallel_resample()


class jittertemp_filter(base_filter):
    def __init__(self, nsteps, noise_shape, n_temp, n_jitt, rho,
                 verbose=False):
        self.nsteps = nsteps
        self.noise_shape = noise_shape
        self.n_temp = n_temp
        self.n_jitt = n_jitt
        self.rho = rho
        self.verbose=verbose

    def assimilation_step(self, y, log_likelihood, ess_tol=0.8):
        N = len(self.ensemble)
        weights = np.zeros(N)
        weights[:] = 1/N
        new_weights = np.zeros(N)
        self.ess_temper = []
        self.theta_temper = []
        W = np.random.randn(N, *(self.noise_shape))
        Wnew = np.zeros(W.shape)

        theta = .0
        while theta <1.: #  Tempering loop
            dtheta = 1.0 - theta
            # forward model step
            for i in range(N):
                # put result of forward model into new_ensemble
                self.model.run(self.nsteps, W[i, :],
                               self.ensemble[i], self.new_ensemble[i])
            ess = 0.
            while ess < ess_tol*N:
                for i in range(N):
                    Y = self.model.obs(self.new_ensemble[i])
                    weights[i] = exp(-dtheta*log_likelihood(y-Y))
                weights /= np.sum(weights)
                ess = 1/np.sum(weights**2)
                if ess < ess_tol*N:
                    dtheta = 0.5*dtheta
            self.ess_temper.append(ess)
            theta += dtheta
            self.theta_temper.append(theta)

            # resampling BEFORE jittering
            s = residual_resampling(weights, model=SimModel)
            self.e_s = s
            if self.verbose:
                print("Updating ensembles")
            for i in range(N):
                self.new_ensemble[i].assign(self.ensemble[s[i]])
                Wnew[i, :] = W[s[i], :]
            for i in range(N):
                self.ensemble[i].assign(self.new_ensemble[i])
                W[i, :] = Wnew[i, :]

            for l in range(self.n_jitt): # Jittering loop
                if self.verbose:
                    print("Jitter, Temper step", l, k)
                # proposal
                Wnew = self.rho*W + (1-self.rho**2)**0.5*np.random.randn(N, *(self.noise_shape))

                # forward model step
                for i in range(N):
                    # put result of forward model into new_ensemble
                    self.model.run(self.nsteps, Wnew[i, :],
                                   self.ensemble[i], self.new_ensemble[i])

                    # particle weights
                    Y = self.model.obs(self.new_ensemble[i])
                    new_weights[i] = exp(-theta*log_likelihood(y-Y))
                    if l == 0:
                        weights[i] = new_weights[i]
                    else:
                        #  Metropolis MCMC
                        p_accept = min(1, new_weights[i]/weights[i])
                        #accept or reject tool
                        if np.random.rand() < p_accept:
                            weights[i] = new_weights[i]
                            W[i,:] = Wnew[i,:]

                weights /= np.sum(weights)
                self.e_weight = weights

        if self.verbose:
            print("Advancing ensemble")
        self.model.run(self.nsteps, W[i, :],
                                   self.ensemble[i], self.ensemble[i])
        if self.verbose:
            print("assimilation step complete")
