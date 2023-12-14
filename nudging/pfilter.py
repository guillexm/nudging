from abc import ABCMeta, abstractmethod, abstractproperty
from firedrake import *
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from .resampling import *
import numpy as np
import pyadjoint
import gc
from .parallel_arrays import DistributedDataLayout1D, SharedArray, OwnedArray
from firedrake_adjoint import *
pyadjoint.tape.pause_annotation()

class base_filter(object, metaclass=ABCMeta):
    ensemble = []
    new_ensemble = []

    def __init__(self):
        pass

    def setup(self, nensemble, model, resampler_seed=34343):
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
        self.proposal_ensemble = []
        for i in range(self.nensemble[self.ensemble_rank]):
            self.ensemble.append(model.allocate())
            self.new_ensemble.append(model.allocate())
            self.proposal_ensemble.append(model.allocate())

        # some numbers for shared array and owned array
        self.nlocal = self.nensemble[self.ensemble_rank]
        self.nglobal = int(np.sum(self.nensemble))

        # Shared array for the potentials
        self.potential_arr = SharedArray(partition=self.nensemble, dtype=float,
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
        self.resampler = residual_resampling(seed=resampler_seed, residual=True)

    def index2rank(self, index):
        for rank in range(len(self.offset_list)):
            if self.offset_list[rank] - index > 0:
                rank -= 1
                break
        return rank

    def parallel_resample(self, dtheta=1):

        self.potential_arr.synchronise(root=0)
        if self.ensemble_rank == 0:
            potentials = self.potential_arr.data()
            # renormalise
            potentials -= np.mean(potentials)
            weights = np.exp(-dtheta*potentials)
            weights /= np.sum(weights)
            self.ess = 1/np.sum(weights**2)
            if self.verbose:
                PETSc.Sys.Print("ESS", self.ess)

        # compute resampling protocol on rank 0
        if self.ensemble_rank == 0:
            s = self.resampler.resample(weights, self.model)
            for i in range(self.nglobal):
                self.s_arr[i]=s[i]

        # broadcast protocol to every rank
        self.s_arr.synchronise()
        s_copy = self.s_arr.data()
        self.s_copy = s_copy

        mpi_requests = []

        for ilocal in range(self.nensemble[self.ensemble_rank]):
            iglobal = self.layout.transform_index(ilocal, itype='l',
                                             rtype='g')
            # add to send list
            targets = []
            for j in range(self.s_arr.size):
                if s_copy[j] == iglobal:
                    targets.append(j)

            for target in targets:
                if type(self.ensemble[ilocal] == 'list'):
                    for k in range(len(self.ensemble[ilocal])):
                        request_send = self.subcommunicators.isend(
                            self.ensemble[ilocal][k],
                            dest=self.index2rank(target),
                            tag=1000*target+k)
                        mpi_requests.extend(request_send)
                else:
                    request_send = self.subcommunicators.isend(
                        self.ensemble[ilocal],
                        dest=self.index2rank(target),
                        tag=target)
                    mpi_requests.extend(request_send)

            source_rank = self.index2rank(s_copy[iglobal])
            if type(self.ensemble[ilocal] == 'list'):
                for k in range(len(self.ensemble[ilocal])):
                    request_recv = self.subcommunicators.irecv(
                        self.new_ensemble[ilocal][k],
                        source=source_rank,
                        tag=1000*iglobal+k)
                    mpi_requests.extend(request_recv)
            else:
                request_recv = self.subcommunicators.irecv(
                    self.new_ensemble[ilocal],
                    source=source_rank,
                    tag=iglobal)
                mpi_requests.extend(request_recv)

        MPI.Request.Waitall(mpi_requests)
        for i in range(self.nlocal):
            for j in range(len(self.ensemble[i])):
                self.ensemble[i][j].assign(self.new_ensemble[i][j])




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
        for i in range(self.nensemble[self.ensemble_rank]):
            # set the particle value to the global index
            self.ensemble[i].assign(self.offset_list[self.ensemble_rank]+i)

            Y = self.model.obs()
            self.potential_arr.dlocal[i] = assemble(log_likelihood(y,Y))
        self.parallel_resample()

class bootstrap_filter(base_filter):

    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose


    def assimilation_step(self, y, log_likelihood):
        N = self.nensemble[self.ensemble_rank]
        # forward model step
        for i in range(N):
            self.model.randomize(self.ensemble[i])
            self.model.run(self.ensemble[i], self.ensemble[i])

            Y = self.model.obs()
            self.potential_arr.dlocal[i] = assemble(log_likelihood(y,Y))
        self.parallel_resample()


class jittertemp_filter(base_filter):
    def __init__(self, n_jitt, delta=None,
                 verbose=False, MALA=False, nudging=False,
                 visualise_tape=False):
        self.delta = delta
        self.verbose=verbose
        self.MALA = MALA
        self.model_taped = False
        self.nudging = nudging
        self.visualise_tape = visualise_tape
        self.n_jitt = n_jitt

        if MALA:
           PETSc.Sys.Print("Warning, we are not currently computing the Metropolis correction for MALA. Choose a small delta.")

    def setup(self, nensemble, model, resampler_seed=34343):
        super(jittertemp_filter, self).setup(
            nensemble, model, resampler_seed=resampler_seed)
        # Owned array for sending dtheta
        self.dtheta_arr = OwnedArray(size = self.nglobal, dtype=float,
                                     comm=self.subcommunicators.ensemble_comm,
                                     owner=0)

    def adaptive_dtheta(self, dtheta, theta, ess_tol):
        N = self.nensemble[self.ensemble_rank]
        self.potential_arr.synchronise(root=0)
        if self.ensemble_rank == 0:
            potentials = self.potential_arr.data()
            ess =0.
            while ess < ess_tol*sum(self.nensemble):
                # renormalise using dtheta
                potentials -= np.mean(potentials)
                weights = np.exp(-dtheta*potentials)
                weights /= np.sum(weights)
                ess = 1/np.sum(weights**2)
                #PETSc.Sys.Print('ESS, Dtheta', ess, dtheta)
                if ess < ess_tol*sum(self.nensemble):
                    dtheta = 0.5*dtheta

            # abuse owned array to broadcast dtheta
            for i in range(self.nglobal):
                self.dtheta_arr[i]=dtheta

        # broadcast dtheta to every rank
        self.dtheta_arr.synchronise()
        dtheta = self.dtheta_arr.data()[0]
        theta += dtheta
        return dtheta

    def assimilation_step(self, y, log_likelihood, ess_tol=0.8):
        N = self.nensemble[self.ensemble_rank]
        potentials = np.zeros(N)
        new_potentials = np.zeros(N)
        self.ess_temper = []
        self.theta_temper = []
        nsteps = self.model.nsteps

        self.temp_count  = 0
        # tape the forward model
        if not self.model_taped:
            if self.verbose:
                PETSc.Sys.Print("taping forward model")
            self.model_taped = True
            pyadjoint.tape.continue_annotation()
            self.model.run(self.ensemble[0],
                           self.new_ensemble[0])
            #set the controls
            if type(y == Function):
                m = self.model.controls() + [Control(y)]
            else:
                m = self.model.controls()
            #requires log_likelihood to return symbolic
            Y = self.model.obs()
            MALA_J = assemble(log_likelihood(y,Y))

            if self.nudging:
                nudge_J = assemble(log_likelihood(y,Y))
                nudge_J += self.model.lambda_functional()
                # set up the functionals
                # functional for nudging
                self.Jhat = []
                for step in range(nsteps+1, nsteps*2+1):
                    # 0 component is state
                    # 1 .. step is noise
                    # step + 1 .. 2*step is lambdas
                    assert(self.model.lambdas)
                    # we only update lambdas[step] on timestep step
                    components = [step]

                    self.Jhat.append(ReducedFunctional(nudge_J, m,
                                                  derivative_components=
                                                  components))
            #functional for MALA
            components = [j for j in range(1, nsteps+1)]
            self.Jhat_dW = ReducedFunctional(MALA_J, m,
                                        derivative_components=
                                        components)
            if self.visualise_tape:
                tape = pyadjoint.get_working_tape()
                tape.visualise_pdf("t.pdf")
            pyadjoint.tape.pause_annotation()

        if self.nudging:
            if self.verbose:
                PETSc.Sys.Print("Starting nudging")
            for i in range(N):
                # zero the noise and lambdas in preparation for nudging
                for step in range(nsteps):
                    self.ensemble[i][step+1].assign(0.) # the noise
                    self.ensemble[i][nsteps+step+1].assign(0.) # the nudging
                # nudging one step at a time
                for step in range(nsteps):
                    PETSc.garbage_cleanup(PETSc.COMM_SELF)
                    # update with current noise and lambda values
                    self.Jhat[step](self.ensemble[i]+[y])
                    # get the minimum over current lambda
                    if self.verbose:
                        PETSc.Sys.Print("Solving for Lambda step ", step,
                                        "local ensemble member ", i)
                    self.Jhat[step].derivative()
                    #print(gh(self.ensemble[i]))
                    if i ==0:
                        Xopt = minimize(self.Jhat[step], options={"disp": False})
                    else:
                        Xopt = minimize(self.Jhat[step])

                    # Xopt = minimize(self.Jhat[step], options={"disp": False, "gtol": 0.01})
                    ##Xopt = minimize(self.Jhat[step], options={"disp": True, "maxiter": 5,  "gtol": 0.1})
                    # place the optimal value of lambda into ensemble
                    self.ensemble[i][nsteps+1+step].assign(
                        Xopt[nsteps+1+step])
                    # get the randomised noise for this step
                    self.model.randomize(Xopt) # not efficient!
                    # just copy in the current component
                    self.ensemble[i][1+step].assign(Xopt[1+step])
        else:
            for i in range(N):
                # generate the initial noise variables
                self.model.randomize(self.ensemble[i])

        # Compute initial potentials
        for i in range(N):
            # put result of forward model into new_ensemble
            self.model.run(self.ensemble[i], self.new_ensemble[i])
            Y = self.model.obs()
            self.potential_arr.dlocal[i] = assemble(log_likelihood(y,Y))

        theta = .0
        while theta <1.: #  Tempering loop
            dtheta = 1.0 - theta
            self.temp_count += 1

            # adaptive dtheta choice
            dtheta = self.adaptive_dtheta(dtheta, theta,  ess_tol)
            theta += dtheta
            self.theta_temper.append(theta)
            if self.verbose:
                PETSc.Sys.Print("theta", theta, "dtheta", dtheta)

            # resampling BEFORE jittering
            self.parallel_resample(dtheta)

            for l in range(self.n_jitt): # Jittering loop
                if self.verbose:
                    PETSc.Sys.Print("Jitter, Temper step", l)

                # forward model step
                for i in range(N):
                    if self.MALA:
                        # run the model and get the functional value with
                        # ensemble[i]
                        self.Jhat_dW(self.ensemble[i]+[y])
                        # use the taped model to get the derivative
                        g = self.Jhat_dW.derivative()
                        # proposal
                        self.model.copy(self.ensemble[i],
                                        self.proposal_ensemble[i])
                        delta = self.delta
                        self.model.randomize(self.proposal_ensemble[i],
                                             (
                                                 (2-delta)/(2+delta)),
                                             (
                                                 (8*delta)**0.5/(2+delta)),
                                             gscale=(
                                                 -2*delta/(2+delta)),g=g)
                    else:
                    # proposal PCN
                        self.model.copy(self.ensemble[i],
                                            self.proposal_ensemble[i])
                        delta = self.delta
                        self.model.randomize(self.proposal_ensemble[i],
                                            (2-delta)/(2+delta),
                                            (8*delta)**0.5/(2+delta))
                    # put result of forward model into new_ensemble
                    self.model.run(self.proposal_ensemble[i],
                                   self.new_ensemble[i])

                    # particle potentials
                    Y = self.model.obs()
                    new_potentials[i] = theta*assemble(log_likelihood(y,Y))
                    #accept reject of MALA and Jittering
                    if l == 0:
                        potentials[i] = new_potentials[i]
                    else:
                        # Metropolis MCMC
                        if self.MALA:
                            p_accept = 1
                        else:
                            p_accept = min(1,
                                            exp(potentials[i]
                                                - new_potentials[i]))
                        # accept or reject tool
                        u = self.model.rg.uniform(self.model.R, 0., 1.0)
                        if u.dat.data[:] < p_accept:
                            potentials[i] = new_potentials[i]
                            self.model.copy(self.proposal_ensemble[i],
                                            self.ensemble[i])

        if self.verbose:
            PETSc.Sys.Print("Advancing ensemble")
        for i in range(N):
            self.model.run(self.ensemble[i], self.ensemble[i])
        if self.verbose:
            PETSc.Sys.Print("assimilation step complete")
        # momeory leak fix trick
        PETSc.garbage_cleanup(PETSc.COMM_SELF)
