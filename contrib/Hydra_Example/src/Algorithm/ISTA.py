from cil.optimisation.functions import IndicatorBox, LeastSquares
from cil.optimisation.algorithms.FISTA import ISTA

class ISTA_Algo(object):
    def  __init__(self, name, experiment, function):
        print(name)
        f_gd = function.get_Func()
        initial = experiment.get_init()
        step_size = 1 / f_gd.L
        g_fun = IndicatorBox(lower=0)

        self.ista = ISTA(initial=initial, f=f_gd, g=g_fun,
                            step_size=step_size, update_objective_interval=1,
                            max_iteration=1e50)

        
                            

    def get_Algo(self):
        return self.ista