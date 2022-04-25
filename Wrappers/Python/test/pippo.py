from cil.framework import ImageGeometry, BlockDataContainer

a = 2
b = 3
N1 = 2
N2 = 3
ig = ImageGeometry(N1,N2)
x = ig.allocate(1)
y = ig.allocate(2)
bdc1 = BlockDataContainer(2*x, y)
bdc2 = BlockDataContainer(x*0, 0*y) 

bdc3 = bdc1.add(2)
print (bdc3[0].as_array(), bdc3[1].as_array())

bdc1.add(3, out=bdc2)
print (bdc2[0].as_array(), bdc2[1].as_array())


# out = bdc1.sapyb(a,bdc2,b)