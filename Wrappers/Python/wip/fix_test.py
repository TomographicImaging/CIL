import numpy as np
import numpy
from ccpi.optimisation.operators import *
from ccpi.optimisation.algorithms import *
from ccpi.optimisation.functions import *
from ccpi.framework import *

def isSizeCorrect(data1 ,data2):
    if issubclass(type(data1), DataContainer) and \
       issubclass(type(data2), DataContainer):
        # check dimensionality
        if data1.check_dimensions(data2):
            return True
    elif issubclass(type(data1) , numpy.ndarray) and \
         issubclass(type(data2) , numpy.ndarray):
        return data1.shape == data2.shape
    else:
        raise ValueError("{0}: getting two incompatible types: {1} {2}"\
                         .format('Function', type(data1), type(data2)))
    return False

class Norm1(Function):
    
    def __init__(self,gamma):
        super(Norm1, self).__init__()
        self.gamma = gamma
        self.L = 1
        self.sign_x = None
    
    def __call__(self,x,out=None):
        if out is None:
            return self.gamma*(x.abs().sum())
        else:
            if not x.shape == out.shape:
                raise ValueError('Norm1 Incompatible size:',
                                 x.shape, out.shape)
            x.abs(out=out)
            return out.sum() * self.gamma
    
    def prox(self,x,tau):
        return (x.abs() - tau*self.gamma).maximum(0) * x.sign()
    
    def proximal(self, x, tau, out=None):
        if out is None:
            return self.prox(x, tau)
        else:
            if isSizeCorrect(x,out):
                # check dimensionality
                if issubclass(type(out), DataContainer):
                    v = (x.abs() - tau*self.gamma).maximum(0)
                    x.sign(out=out)
                    out *= v
                    #out.fill(self.prox(x,tau))    
                elif issubclass(type(out) , numpy.ndarray):
                    v = (x.abs() - tau*self.gamma).maximum(0)
                    out[:] = x.sign()
                    out *= v
                    #out[:] = self.prox(x,tau)
            else:
                raise ValueError ('Wrong size: x{0} out{1}'.format(x.shape,out.shape) )

opt = {'memopt': True}
# Problem data.
m = 500
n = 200

# if m < n then the problem is under-determined and algorithms will struggle to find a solution. 
# One approach is to add regularisation

#np.random.seed(1)
Amat = np.asarray( np.random.randn(m, n), dtype=numpy.float32)
Amat = np.asarray( np.random.random_integers(1,10, (m, n)), dtype=numpy.float32)
#Amat = np.asarray(np.eye(m), dtype=np.float32) * 2
A = LinearOperatorMatrix(Amat)
bmat = np.asarray( np.random.randn(m), dtype=numpy.float32)
#bmat *= 0 
#bmat += 2
print ("bmat", bmat.shape)
print ("A", A.A)
#bmat.shape = (bmat.shape[0], 1)

# A = Identity()
# Change n to equal to m.
vgb = VectorGeometry(m)
vgx = VectorGeometry(n)
b = vgb.allocate(VectorGeometry.RANDOM_INT, dtype=numpy.float32)
# b.fill(bmat)
#b = DataContainer(bmat)

# Regularization parameter
lam = 10
opt = {'memopt': True}
# Create object instances with the test data A and b.
f = Norm2Sq(A, b, c=1., memopt=True)
#f = FunctionOperatorComposition(A, L2NormSquared(b=bmat))
g0 = ZeroFunction()

#f.L = 30.003
x_init = vgx.allocate(VectorGeometry.RANDOM, dtype=numpy.float32)
x_initcgls = x_init.copy()

a = VectorData(x_init.as_array(), deep_copy=True)

assert id(x_init.as_array()) != id(a.as_array())


#f.L = LinearOperator.PowerMethod(A, 25, x_init)[0] 
#print ('f.L', f.L)
rate = (1 / f.L) / 6
#f.L *= 12

# Initial guess
#x_init = DataContainer(np.zeros((n, 1)))
print ('x_init', x_init.as_array())
print ('b', b.as_array())
# Create 1-norm object instance
g1_new = lam * L1Norm()
g1 = Norm1(lam)

g1 = ZeroFunction()
#g1(x_init) 
x = g1.prox(x_init, 1/f.L ) 
print ("g1.proximal ", x.as_array())

x = g1.prox(x_init, 0.03 ) 
print ("g1.proximal ", x.as_array())
x = g1_new.proximal(x_init, 0.03 ) 
print ("g1.proximal ", x.as_array())

x1 = vgx.allocate(VectorGeometry.RANDOM, dtype=numpy.float32)
pippo = vgx.allocate()

print ("x_init", x_init.as_array())
print ("x1", x1.as_array())
a = x_init.subtract(x1, out=pippo)

print ("pippo", pippo.as_array())
print ("x_init", x_init.as_array())
print ("x1", x1.as_array())

y = A.direct(x_init)
y *= 0
A.direct(x_init, out=y)

# Combine with least squares and solve using generic FISTA implementation
#x_fista1, it1, timing1, criter1 = FISTA(x_init, f, g1, opt=opt)
def callback(it,  objective, solution):
    print ("callback " , it , objective, f(solution))

fa = FISTA(x_init=x_init, f=f, g=g1)
fa.max_iteration = 1000
fa.update_objective_interval = int( fa.max_iteration / 10 ) 
fa.run(fa.max_iteration, callback = None, verbose=True)

gd = GradientDescent(x_init=x_init, objective_function=f, rate = rate )
gd.max_iteration = 5000
gd.update_objective_interval = int( gd.max_iteration / 10 ) 
gd.run(gd.max_iteration, callback = None, verbose=True)



cgls = CGLS(x_init= x_initcgls, operator=A, data=b)
cgls.max_iteration = 1000
cgls.update_objective_interval = int( cgls.max_iteration / 10 ) 

#cgls.should_stop = stop_criterion(cgls)
cgls.run(cgls.max_iteration, callback = callback, verbose=True)



# Print for comparison
print("FISTA least squares plus 1-norm solution and objective value:")
print(fa.get_output().as_array())
print(fa.get_last_objective())

print ("data           ", b.as_array())
print ('FISTA          ', A.direct(fa.get_output()).as_array())
print ('GradientDescent', A.direct(gd.get_output()).as_array())
print ('CGLS           ', A.direct(cgls.get_output()).as_array())

cond = numpy.linalg.cond(A.A)

print ("cond" , cond)

#%%
try:
    import cvxpy as cp
    # Construct the problem.
    x = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(A.A*x - bmat))
    prob = cp.Problem(objective)
    # The optimal objective is returned by prob.solve().
    result = prob.solve(solver = cp.MOSEK)

    print ('CGLS           ', cgls.get_output().as_array())
    print ('CVX           ', x.value)

    print ('FISTA           ', fa.get_output().as_array())
    print ('GD           ', gd.get_output().as_array())
except ImportError as ir:
    pass

    #%%





