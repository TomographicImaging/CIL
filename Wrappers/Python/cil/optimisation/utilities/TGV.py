# set up TGV
from cil.optimisation.functions import MixedL21Norm, BlockFunction, L2NormSquared
from cil.optimisation.operators import BlockOperator, IdentityOperator, GradientOperator, \
    SymmetrisedGradientOperator, ZeroOperator

def setup_explicit_TGV(A, data, alpha, delta=1.0):
    '''Class to setup LS + TGV problem with explicit PDHG

    Parameters
    ----------
    A : ProjectionOperator
        Forward operator.
    data : AcquisitionData
    alpha : float
        Regularisation parameter.
    delta : float, default 1.0
        The Regularisation parameter for the symmetrised gradient, beta, can be controlled by delta
        with beta = delta * alpha.       
    '''

    # delta = beta / alpha
    # beta =  alpha * delta
    
    beta = alpha * delta
    
    f1 = 0.5 * L2NormSquared(b=data)
    f2 = MixedL21Norm()
    f3 = MixedL21Norm() 
    F = BlockFunction(f1, f2, f3)         

    # Define BlockOperator K
                
    # Set up the 3 operator A, Grad and Epsilon                           
    # A, the projection operator is passed by the user    
    K11 = A
    grad = GradientOperator(K11.domain)
    K21 = alpha * grad
    # https://tomographicimaging.github.io/CIL/nightly/optimisation.html#cil.optimisation.operators.SymmetrisedGradientOperator
    K32 = beta * SymmetrisedGradientOperator(K21.range)
    # these define the domain and range of the other operators
    K12 = ZeroOperator(K32.domain, K11.range)
    K22 = -alpha * IdentityOperator(domain_geometry=K21.range, range_geometry=K32.range)
    K31 = ZeroOperator(K11.domain, K32.range)

    K = BlockOperator(K11, K12, K21, K22, K31, K32, shape=(3,2) )

    return K, F


def setup_explicit_TGV_stochastic(A, data, alpha, delta=1.0):
    '''Class to setup LS + TGV problem with explicit PDHG

    Parameters
    ----------
    A : ProjectionOperator
        Forward operator.
    data : AcquisitionData
    alpha : float
        Regularisation parameter.
    delta : float, default 1.0
        The Regularisation parameter for the symmetrised gradient, beta, can be controlled by delta
        with beta = delta * alpha.       
    '''

    # delta = beta / alpha
    # beta =  alpha * delta
    
    beta = alpha * delta
    
    fs = []
    for el in data.containers:
        fs.append(0.5 * L2NormSquared(b=el))

    # f1 = 0.5 * L2NormSquared(b=data)
    f2 = MixedL21Norm()
    f3 = MixedL21Norm() 
    F = BlockFunction(*fs, f2, f3)         

    # Define BlockOperator K
                
    # Set up the 3 operator A, Grad and Epsilon                           
    # A, the projection operator is passed by the user    
    K11 = A
    K21 = alpha * GradientOperator(K11.domain)
    # https://tomographicimaging.github.io/CIL/nightly/optimisation.html#cil.optimisation.operators.SymmetrisedGradientOperator
    K32 = beta * SymmetrisedGradientOperator(K21.range)
    # these define the domain and range of the other operators
    # K12 = ZeroOperator(K32.domain, K11.range)
    K22 = -alpha * IdentityOperator(domain_geometry=K21.range, range_geometry=K32.range)
    K31 = ZeroOperator(K11.domain, K32.range)

    K11i = []
        
    for el in A.operators:
        K12 = ZeroOperator(K32.domain, el.range)
        K11i.append(BlockOperator(el, K12, shape=(1,2)))

    K = BlockOperator(*K11i, BlockOperator(K21, K22, shape=(1,2)), 
                    BlockOperator(K31, K32, shape=(1,2)),
         shape=(len(K11i)+2,1) )

    return K, F