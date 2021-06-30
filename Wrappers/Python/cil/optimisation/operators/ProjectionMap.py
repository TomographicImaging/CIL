from cil.optimisation.operators import LinearOperator
from cil.framework import BlockGeometry



class ProjectionMap(LinearOperator):

    r""" Projection Map or Canonincal Projection (https://en.wikipedia.org/wiki/Projection_(mathematics))
    
    takes an element x = (x_{0},\dots,x_{i},\dots,x_{n}}) from a Cartesian product space X_{1}\times\cdots\times X_{n}\rightarrow X_{i}

    and projects it to element x_{i} specified by the index i.

    .. math:: \pi_{i}: X_{1}\times\cdots\times X_{n}\rightarrow X_{i}

    .. math:: \pi_{i}(x_{0},\dots,x_{i},\dots,x_{n}) = x_{i}

    The adjoint operation, is defined as 

    .. math:: \pi_{i}^{*}(x_{i}) = (0, \cdots, x_{i}, \cdots, 0)

    :param domain_geometry: The domain of the Projection Map. A BlockGeometry is expected
    :type domain_geometry: `BlockGeometry`
    :param index: Index to project to the corresponding ImageGeometry X_{index}
    :type index: int   
    :return: returns a DataContainer 
    :rtype: DataContainer    

    """

    
    def __init__(self, domain_geometry, index, range_geometry=None):
        
        self.index = index

        if not isinstance(domain_geometry, BlockGeometry):
            raise ValueError("BlockGeometry is expected, {} is passed.".format(domain_geometry.__class__.__name__))

        if self.index > len(domain_geometry.geometries):
            raise ValueError("Index = {} is larger than the total number of geometries = {}".format(index, len(domain_geometry.geometries)))

        if range_geometry is None:
            range_geometry = domain_geometry.geometries[self.index]
            
        super(ProjectionMap, self).__init__(domain_geometry=domain_geometry, 
                                           range_geometry=range_geometry)   
        
    def direct(self,x,out=None):
                        
        if out is None:
            return x[self.index]
        else:
            out.fill(x[self.index])
    
    def adjoint(self,x, out=None):
        
        if out is None:
            tmp = self.domain_geometry().allocate()
            tmp[self.index].fill(x)            
            return tmp
        else:
            out[self.index].fill(x) 

if __name__ == '__main__':

    from cil.framework import ImageGeometry, BlockGeometry
    import numpy as np

    print("Check if direct is correct")
    ig1 = ImageGeometry(3,4)
    ig2 = ImageGeometry(5,6)
    ig3 = ImageGeometry(5,6,4)

    bg = BlockGeometry(ig1,ig2, ig3)

    x = bg.allocate('random')
    x0 = x[0]
    x1 = x[1]
    x2 = x[2]


    for i in range(3):

        proj_map = ProjectionMap(bg, i)

        # res1 is in ImageData from the X_{i} "ImageGeometry"
        res1 = proj_map.direct(x)

        # res2 is in ImageData from the X_{i} "ImageGeometry" using out
        res2 = bg.geometries[i].allocate(0)
        proj_map.direct(x, out=res2)

        # Assert with and withou out
        np.testing.assert_array_almost_equal(res1.array, res2.array)

        # Depending on which index is used, check if x0, x1, x2 are the same with res2

        if i==0:            
            np.testing.assert_array_almost_equal(x0.array, res2.array)
        elif i==1: 
            np.testing.assert_array_almost_equal(x1.array, res2.array)   
        elif i==2:
            np.testing.assert_array_almost_equal(x2.array, res2.array)  
        else:
            pass      

    print("Test passed \n")    

    print("Check if adjoint is correct")

    bg = BlockGeometry(ig1, ig2, ig3, ig1, ig2, ig3)

    x = ig1.allocate('random')

    index=3
    proj_map = ProjectionMap(bg, index)

    res1 = bg.allocate(0)
    proj_map.adjoint(x, out=res1)

    # check if all indices return arrays filled with 0, except the input index

    for i in range(len(bg.geometries)):

        if i!=index:
            np.testing.assert_array_almost_equal(res1[i].array, bg.geometries[i].allocate().array)

    print("Test passed \n")       

    print("Check error messages")
    # Check if index is correct wrt length of Cartesian Product
    try:
        ig = ImageGeometry(3,4)
        bg = BlockGeometry(ig,ig)
        index = 3
        proj_map = ProjectionMap(bg, index)
    except ValueError as err:
            print(err) 

    # Check error if an ImageGeometry is passed
    try:
        proj_map = ProjectionMap(ig, index)               
    except ValueError as err:
        print(err)   
    print("Test passed \n")           



    

