from firedrake import *
ensemble = Ensemble(COMM_WORLD, 2)

print(ensemble.ensemble_comm.rank)