from ccpi.framework import ImageGeometry


ok = 0 
nok = 0
vg = ImageGeometry(3,4,5,channels=2)
# print ( [ImageGeometry.CHANNEL, ImageGeometry.VERTICAL,
#     ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X],
#                    vg.dimension_labels)
for i in range(10000):
    ss = vg.allocate()
    #print (i)
    # l = [ss.dimension_labels[i] for i in range(len(ss.dimension_labels.keys()))]
    # if l != vg.dimension_labels:
    #     print ("Error:\n{}{}{}".format(l , "\n", vg.dimension_labels))
    ss2 = ss.subset(vertical = 0, channel=0)


#print([ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X], ss2.geometry.dimension_labels)
    try:
        assert [ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X] == ss2.geometry.dimension_labels
        ok += 1
    except AssertionError as ae:
        nok += 1
print ("OK {}  NOK {}".format(ok, nok))