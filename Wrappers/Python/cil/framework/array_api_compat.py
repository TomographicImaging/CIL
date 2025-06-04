# Updates to array API compatibility layer for CIL
from array_api_compat import array_namespace

def expand_dims(array, axis):
    '''Expand dimensions of an array along specified axes.
    
    Parameters
    ----------
    array : array-like
        The input array to expand.
    axis : int or tuple of int
        The axis or axes along which to expand the dimensions.
        
    Returns
    -------
    array-like
        The array with expanded dimensions. It may be a new array or the same array with expanded dimensions.

    Raises:
    --------
    IndexError If provided an invalid axis position, an IndexError should be raised.
    
    Notes:
    This function recursively expands the dimensions of the input array along the specified axes if a list or tuple of ints is provided.
    '''
    xp = array_namespace(array)
    
    if isinstance(axis, int):
        return xp.expand_dims(array, axis=axis)
    axis = list(axis)
    ax = axis.pop(0)
    if len(axis) == 1:
        axis = axis[0]
    return expand_dims(xp.expand_dims(array, axis=ax), axis=axis)

def squeeze(array, axis=None):
    '''squeezes the array, removing all singleton dimensions recursively
    
    Parameters
    ----------
    array : array-like
        The array to squeeze
    axis : int or tuple of int, optional
        The axis or axes to squeeze. If None, all singleton dimensions are removed.

    Returns
    -------
    array-like
        The squeezed array with all singleton dimensions removed. If the input array has no singleton dimensions, it is returned unchanged.
    '''
    xp = array_namespace(array)
    # find and remove singleton dimensions
    if axis is None:
        s = xp.nonzero(xp.asarray(array.shape) == 1)[0]
        axis = s.tolist()
    if isinstance(axis, int):
        return xp.squeeze(array, axis=axis)
    # process from the largest axis to the smallest
    axis = list(axis)
    axis.sort(reverse=True)
    ax = axis.pop(0)
    if len(axis) == 1:
        axis = axis[0]
    return squeeze(xp.squeeze(array, axis=ax), axis=axis)