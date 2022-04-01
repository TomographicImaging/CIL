#%%
from cil.optimisation.functions import MixedL21Norm
from cil.optimisation.operators import GradientOperator
from cil.framework import ImageGeometry
from cil.utilities import dataexample
from cil.utilities.display import show2D
from cil.io import NEXUSDataReader
from cil.processors import Slicer

import time
import os
from tqdm import tqdm
import numpy as np
try:
    from cil.plugins import TomoPhantom
    has_tomophantom = True
except ImportError:
    has_tomophantom = False
has_numba = True
try:
    import numba
except ImportError:
    has_numba = False
    

@numba.jit(nopython=True)
def some_calc(arr, abstau):
    tmp = arr.ravel()
    for i in numba.prange(tmp.size):
        if tmp[i] == 0:
            continue
        el = tmp[i] / abstau
        el -= 1.
        if el <= 0.0:
            el = 0.
        
        tmp[i] = el / tmp[i]
    return arr
    # tmp /= np.abs(tau)
    # # res = (tmp - 1).maximum(0.0) * x/tmp
    # res = tmp - 1
    # res.maximum(0.0, out=res)
    # res /= tmp

    # resarray = res.as_array()
    # resarray[np.isnan(resarray)] = 0
    # res.fill(resarray)

def proximal_new(self, x, tau, out=None):
    
    r"""Returns the value of the proximal operator of the MixedL21Norm function at x.
    
    .. math :: \mathrm{prox}_{\tau F}(x) = \frac{x}{\|x\|_{2}}\max\{ \|x\|_{2} - \tau, 0 \}
    
    where the convention 0 · (0/0) = 0 is used.
    
    """

    # Note: we divide x/tau so the cases of both scalar and 
    # datacontainers of tau to be able to run
    
    tmp = x.pnorm(2)
    if has_numba:
        resarray = some_calc(np.asarray(tmp.as_array(), order='C', dtype=np.float32), np.abs(tau))
        res = tmp 
        res.fill(resarray)
    else:
        tmp /= np.abs(tau, dtype=np.float32)
        # res = (tmp - 1).maximum(0.0) * x/tmp
        res = tmp - 1
        res.maximum(0.0, out=res)
        res /= tmp

        resarray = res.as_array()
        resarray[np.isnan(resarray)] = 0
        res.fill(resarray)
    
    if out is None:
        res = x.multiply(res)
    else:
        x.multiply(res, out = out)
        res = out
    
    
    # # TODO avoid using numpy, add operation in the framework
    # # This will be useful when we add cupy                         
    # for el in res.containers:
    #     elarray = el.as_array()
    #     elarray[np.isnan(elarray)] = 0
    #     el.fill(elarray)

    if out is None:
        return res
def proximal_conjugate_new(self, x, tau, out = None):
    
    r"""Returns the proximal operator of the convex conjugate of function :math:`\tau F` at :math:`x^{*}`
    
    .. math:: \mathrm{prox}_{\tau F^{*}}(x^{*}) = \underset{z^{*}}{\mathrm{argmin}} \frac{1}{2}\|z^{*} - x^{*}\|^{2} + \tau F^{*}(z^{*})
    
    Due to Moreau’s identity, we have an analytic formula to compute the proximal operator of the convex conjugate :math:`F^{*}`
    
    .. math:: \mathrm{prox}_{\tau F^{*}}(x) = x - \tau\mathrm{prox}_{\tau^{-1} F}(\tau^{-1}x)
            
    """
    try:
        tmp = x
        x.divide(tau, out = tmp)
    except TypeError:
        tmp = x.divide(tau, dtype=np.float32)

    if out is None:
        val = self.proximal_new(tmp, 1.0/tau)
    else:            
        self.proximal_new(tmp, 1.0/tau, out = out)
        val = out
                
    if id(tmp) == id(x):
        x.multiply(tau, out = x)

    # CIL issue #1078, cannot use axpby
    # val.axpby(-tau, 1.0, x, out=val)
    val.multiply(-tau, out = val)
    val.add(x, out = val)

    if out is None:
        return val
if __name__ == '__main__':
    
    repeat = 10

    # data = dataexample.CAMERA.get((128,128))
    # data = dataexample.SIMULATED_SPHERE_VOLUME.get()

    # N = 512
    # ig = ImageGeometry(N, N, N)
    # data = TomoPhantom.get_ImageData(1, ig)

    reader = NEXUSDataReader()
    fname = os.path.abspath('C:/Users/ofn77899/Data/CTMeeting2022/small_normSPDHG_eTV_alpha_0.0003_it_1260.nxs')
    reader.set_up(file_name=fname)
    data = reader.read()
    data = data.get_slice(vertical='centre')

    roi = {'horizontal_x':(121, 164, 1),
           'horizontal_y':(83,117,1)}
    data = Slicer(roi=roi)(data)

    print (data.geometry)

    setattr(MixedL21Norm, 'proximal_new', proximal_new)
    setattr(MixedL21Norm, 'proximal_conjugate_new', proximal_conjugate_new)
    f = MixedL21Norm()
    # f.proximal_new = proximal_new
    # f.proximal_conjugate_new = proximal_conjugate_new
    

    g = GradientOperator(data.geometry)

    y = g.direct(data)
#%%
    gamma = 1.
    dt1 = []

    ##############
    for i in tqdm(range(repeat)):
        t0 = time.time()
        # current_impl = f.proximal_conjugate(y, gamma)
        # current_impl = f.proximal(y, gamma)
        tmp = y.pnorm(2)
    
        res1 = some_calc(np.asarray(tmp.as_array(), order='C', dtype=np.float32), np.abs(gamma))
        
        t1 = time.time()
        dt1.append(t1-t0)
    
    dt = [sum(dt1)/repeat]

    dt2 = []
    for i in tqdm(range(repeat)):
        t0 = time.time()
        # new_impl = proximal_conjugate_new(f, y, gamma)
        # new_impl = proximal_new(f, y, gamma)
        
        tmp = y.pnorm(2)
        tmp /= np.abs(gamma, dtype=np.float32)
        # res = (tmp - 1).maximum(0.0) * x/tmp
        res = tmp - 1
        res.maximum(0.0, out=res)
        res /= tmp

        resarray = res.as_array()
        resarray[np.isnan(resarray)] = 0
        res.fill(resarray)
        res2=res
        t1 = time.time()
        dt2.append(t1-t0)

#%%    
    title = ['new', 'new numba']

    ratio = (res2/res1).as_array()
    ratio[np.isnan(ratio)] = 0
    show2D([res2, res1], 
            title=title, origin='upper-left', cmap='terrain', fix_range=False)
    show2D([res2-res1, ratio], title=['diff', 'ratio {} {}'.format((ratio).sum(),res2.as_array().size)], 
            cmap='seismic', fix_range=False)

    ##############
#%%
    for i in tqdm(range(repeat)):
        t0 = time.time()
        # current_impl = f.proximal_conjugate(y, gamma)
        current_impl = f.proximal(y, gamma)
        t1 = time.time()
        dt1.append(t1-t0)
    
    dt = [sum(dt1)/repeat]

    dt2 = []
    for i in tqdm(range(repeat)):
        t0 = time.time()
        # new_impl = proximal_conjugate_new(f, y, gamma)
        new_impl = proximal_new(f, y, gamma)
        t1 = time.time()
        dt2.append(t1-t0)
    
    dt.append(sum(dt2)/repeat)
    
    title = ['current', 'current', 'new', 'new', 'data']
    show2D([current_impl[0], current_impl[1], new_impl[0], new_impl[1] , data], 
            title=title, origin='upper-left', cmap='terrain', fix_range=False)
    show2D([current_impl[0]- new_impl[0], current_impl[1] -new_impl[1] , data], 
            cmap='seismic', fix_range=(-15,15))

    for i in range(len(current_impl)):
        np.testing.assert_allclose(current_impl[i].as_array(), new_impl[i].as_array()/2, atol=1e-5, rtol=1e-6 )

    current_impl *= 0
    new_impl *=0

    dt3 = []
    for i in tqdm(range(repeat)):
        t0 = time.time()
        f.proximal_conjugate(y, gamma, out=current_impl)
        t1 = time.time()
        dt3.append(t1-t0)
    
    dt.append(sum(dt3)/repeat)

    dt4 = []
    for i in tqdm(range(repeat)):
        t0 = time.time()
        proximal_conjugate_new(f, y, gamma, out=new_impl)
        t1 = time.time()
        dt4.append(t1-t0)
    
    dt.append(sum(dt4)/repeat)
    for i in range(len(current_impl)):
        np.testing.assert_allclose(current_impl[i].as_array(), new_impl[i].as_array(), atol=1e-5, rtol=1e-6 )

    print (dt)
    # # http://proximity-operator.net/proximityoperator.html
    # # prox conj
    # dt2 = []
    # y = g.direct(data)
    # proximal = f.proximal(y, 1/gamma)
    # for i in tqdm(range(repeat)):
    #     t0 = time.time()
    #     y = g.direct(data)
    #     # scaled_data = y * (1/gamma)
    #     # proximal = f.proximal(scaled_data, 1/gamma)
    #     # proximal.sapyb(-1/gamma, y, 1., out=proximal)
    #     y *= 1/gamma
    #     f.proximal(y, 1/gamma, out=proximal)
    #     proximal.sapyb(-1/gamma, y, gamma, out=proximal)
    #     y *= gamma
    #     t1 = time.time()
    #     dt2.append(t1-t0)
    # dt.append(sum(dt2)/repeat)
    # print (dt)

    show2D([current_impl[0], current_impl[1], new_impl[0], new_impl[1] , data], 
            title=title, origin='upper-left', cmap='terrain', fix_range=False)
    show2D([current_impl[0]- new_impl[0], current_impl[1] -new_impl[1] , data], 
            cmap='seismic', fix_range=(-15,15))




# %%
