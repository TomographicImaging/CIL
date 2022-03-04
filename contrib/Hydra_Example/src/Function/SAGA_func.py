from cil.optimisation.functions import LeastSquares

class Leastsquares(object):
    def  __init__(self, name, experiment):
        self.f_gd = LeastSquares(experiment.get_Fwd_Op(), experiment.get_data())
        
    def get_Func(self):
        return self.f_gd