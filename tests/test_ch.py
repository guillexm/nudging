from firedrake import *
from nudging import *
import numpy as np

def test_ch():
    model = Camsholm(10)
    In = model.allocate()
    Out = model.allocate()

    x, = SpatialCoordinate(model.mesh)
    _, u0 = In.split()
    u0.interpolate(0.2*2/(exp(x-403./15.) + exp(-x+403./15.))
                   + 0.5*2/(exp(x-203./15.)+exp(-x+203./15.)))

    nsteps = 10
    W = np.random.randn(nsteps, 4)
    model.run(nsteps, W, In, Out)
    model.obs(Out)
