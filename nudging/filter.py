from abc import ABCMeta, abstractmethod, abstractproperty
class base_filter(object, metaclass=ABCMeta):
    ensemble = []

    def __init__(self):
        pass

    def setup(self, nensemble, model)
        """
        Construct the ensemble

        nensemble - number of ensemble members
        model - the model to use
        """
        self.model = model
        for i in range(nensemble):
            self.ensemble.append(model.allocate())

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
