from abc import ABCMeta, abstractmethod, abstractproperty
class model(object, metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def set_initial_state(X):
        pass

    @abstractmethod
    def set_noise(W):
        pass

    @abstractmethod
    def run(nsteps):
        pass

    
