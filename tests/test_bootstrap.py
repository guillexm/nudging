from firedrake import *
from nudging import *
import numpy as np

def test_bootstrap():
    # create some synthetic data
    model = Camsholm(10)
    X_truth = model.allocate()
    _, u0 = X_truth.split()
    u0.interpolate(0.2*2/(exp(x-403./15.) + exp(-x+403./15.))
                   + 0.5*2/(exp(x-203./15.)+exp(-x+203./15.)))

    nsteps = 5
    W_truth = np.random.randn(nsteps, 4)
    model.run(nsteps, W_truth, X_truth, X_truth)
    y = model.obs(X_truth)

    # bootstrap filter
    bfilter = bootstrap_filter(5, (5, 2))
    nensemble = 10
    bfilter.setup(nensemble, model)

    # initialise ensemble
    bfilter.setup(nensemble, model)

    for i in range(nensemble):
        dx0 = Constant(np.random.randn()*0.1)
        dx1 = Constant(np.random.randn()*0.1)
        a = Constant(np.random.rand())
        b = Constant(np.random.rand())
        u0_exp = a*0.2*2/(exp(x-403./15. + dx0) + exp(-x+403./15. + dx0)) \
            + b*0.5*2/(exp(x-203./15. + dx1)+exp(-x+203./15. + dx1))
        bfilter.ensemble(i).interpolate(u0_exp)

    # do one assimiliation step
    def log_likelihood(dY):
        return np.dot(dY, dY)/0.1**2
    bfilter(y, log_likelihood)
