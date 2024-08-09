# set up TGV
from cil.optimisation.functions import MixedL21Norm, BlockFunction, L2NormSquared, ScaledFunction
from cil.optimisation.operators import BlockOperator, IdentityOperator, GradientOperator, \
    SymmetrisedGradientOperator, ZeroOperator

def setup_explicit_TGV(A, data, alpha, delta=1.0, omega=1):
    '''Function to setup LS + TGV problem for use with explicit PDHG

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
    omega : float, default 1.0
        The constant in front of the data fitting term. Mathematicians like it to be 1/2 but it is 1 by default, 
        i.e. it is ignored if it is 1.

    Returns:
    --------
    K : BlockOperator
    F : BlockFunction
    '''

    # delta = beta / alpha
    # beta =  alpha * delta
    beta = alpha * delta
    f1 = L2NormSquared(b=data)
    if omega != 1:
        f1 = omega * f1
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


def setup_explicit_TV(A, data, alpha, omega=1):
    '''Function to setup LS + TV problem for use with explicit PDHG

    Parameters
    ----------
    A : ProjectionOperator
        Forward operator.
    data : AcquisitionData
    alpha : float
        Regularisation parameter.
    omega : float, default 1.0
        The constant in front of the data fitting term. Mathematicians like it to be 1/2 but it is 1 by default, 
        i.e. it is ignored if it is 1.

    Returns:
    --------
    K : BlockOperator
    F : BlockFunction

    '''

    f1 = L2NormSquared(b=data)
    if omega != 1:
        f1 = omega * f1
    f2 = MixedL21Norm()
    F = BlockFunction(f1, f2)         

    # Define BlockOperator K
                
    K11 = A
    grad = GradientOperator(K11.domain)
    K21 = alpha * grad
    
    K = BlockOperator(K11, K21)

    return K, F
