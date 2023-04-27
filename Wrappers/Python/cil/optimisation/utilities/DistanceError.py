from numbers import Number

class DistanceError():
    
    def __init__(self, algorithm, tolerance=0):
        
        self.algorithm = algorithm
        self.tolerance = tolerance
        
        if isinstance(self.tolerance, Number):
            if self.tolerance<0:
                raise ValuerError(" Tolerance should be large or equal to 0, {} is passed".format(self.tolerance))
        else:
            raise ValuerError(" Tolerance should be positive real number, {} is passed".format(self.tolerance))
            
        #override stopping rule of algorithm class
        self.algorithm.should_stop = self._should_stop        
        self.should_stop = False
        
    def _should_stop(self):

        return self.should_stop 
    
    def __call__(self):
        
        raise NotImplementedError()
        
        
    
class OptimalityDistance(DistanceError): 
    
    
    def __init__(self, algorithm, optimal_solution, tolerance=0):
        
        
        super(OptimalityDistance, self).__init__(algorithm, tolerance=tolerance)
        
        # optimal solution
        self.optimal_solution = optimal_solution

        # store distance from optimal in algorithm
        self.algorithm.optimality_distance = []
        
    def __call__(self):
        
        diff_from_optimal = (self.algorithm.x - self.optimal_solution).norm()
        self.algorithm.optimality_distance.append(diff_from_optimal)

        if diff_from_optimal<self.tolerance:
            self.should_stop = True    
            
class ConsecutiveIterationsDistance(DistanceError):

    def __init__(self, algorithm, tolerance=0):

        super(ConsecutiveIterationsDistance, self).__init__(algorithm, tolerance)
        
        # store distance from optimal in algorithm
        self.algorithm.consecutive_iterations = []  
        
    def __call__(self):

        diff_cons_iterations = (self.algorithm.x - self.algorithm.x_old).norm() 
        self.algorithm.consecutive_iterations.append(diff_cons_iterations)
        # if initial=0 then the first difference will be zero, so we skip it
        if self.algorithm.iteration>1:
            if diff_cons_iterations<self.tolerance:
                self.should_stop = True                     
