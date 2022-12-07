from abc import ABCMeta, abstractmethod, abstractproperty
from firedrake import *
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from .resampling import *
import numpy as np
from scipy.special import logsumexp

from parallel_arrays import SharedArray


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
        self.fin_ensemble = np.sum(nensemble)
        self.nspace = int(COMM_WORLD.size/self.fin_ensemble)
        assert(self.nspace*int(self.fin_ensemble) == COMM_WORLD.size)

        self.subcommunicators = Ensemble(COMM_WORLD, self.nspace)
        # model needs to build the mesh in setup
        self.model.setup(self.subcommunicators.comm) # can be removed from here to the model file 
        #nensemble = tuple(nensemble for _ in range(self.subcommunicators.comm.size))
        #self.nensemble = nensemble
        # setting up ensemble 
        self.ensemble_rank = self.subcommunicators.ensemble_comm.rank
        self.ensemble_size = self.subcommunicators.ensemble_comm.size
        for i in range(self.nensemble):
            self.ensemble.append(model.allocate())
            self.new_ensemble.append(model.allocate())


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


class bootstrap_filter(base_filter):
    def __init__(self, nsteps, noise_shape):
        self.nsteps = nsteps
        self.noise_shape = noise_shape

    def setup(self, nensemble, model):
        super().setup(nensemble, model)
        # allocate working memory for resampling
        self.new_ensemble = []
        for i in range(self.nensemble):
            self.new_ensemble.append(self.model.allocate())

    def assimilation_step(self, y, log_likelihood):
        N = len(self.ensemble)
        weights = np.zeros(N)

        #Shared array
        share_arr = SharedArray(self, dtype=int, comm=self.subcommunicators.ensemble_comm)
        weight_arr = SharedArray(self, dtype=float, comm=self.subcommunicators.ensemble_comm)
       # forward model step
        for i in range(N):
            W = np.random.randn(*(self.noise_shape))
            self.model.run(self.nsteps, W,
                             self.ensemble[i], self.ensemble[i])   # solving FEM with ensemble as input and final sol ensemble

            # particle weights
            Y = self.model.obs(self.ensemble[i])
            weight_arr.dlocal[i] = log_likelihood(y-Y)

        # Synchronising weights across all ranks
        weights = weight_arr.synchronise()

        weights = weight_arr.data()
        # renormalise
        weights = np.exp(-weights)
        weights /= np.sum(weights)

        
        self.ess = 1/np.sum(weights**2)

        s = residual_resampling(weights)

        for i in range(N): # update using dlocal, dglobal
            self.new_ensemble[share_arr.dlocal[i]].assign(self.ensemble[share_arr.dglobal[s[i]]])
        for i in range(N):
            self.ensemble[share_arr.dlocal[i]].assign(self.new_ensemble[share_arr.dlocal[i]])
        

class jittertemp_filter(base_filter):
    def __init__(self, nsteps, noise_shape, n_temp, n_jitt, rho,
                 verbose=False):
        self.nsteps = nsteps
        self.noise_shape = noise_shape
        self.n_temp = n_temp
        self.n_jitt = n_jitt
        self.rho = rho
        self.verbose=verbose

    def setup(self, nensemble, model):
        super().setup(nensemble, model)
        # allocate working memory for resampling and forward model
        self.new_ensemble = []
        for i in range(self.nensemble):
            self.new_ensemble.append(self.model.allocate())

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
            s = residual_resampling(weights)
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
