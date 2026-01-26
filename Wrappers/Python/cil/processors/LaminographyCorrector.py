# %%
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter, sobel
import matplotlib.pyplot as plt
import numpy as np

from cil.framework import Processor
from cil.plugins.astra.operators import ProjectionOperator
from cil.plugins.astra.processors import FBP
from cil.processors import Binner, Slicer
from cil.framework import AcquisitionData
from cil.framework.labels import AcquisitionType, AcquisitionDimension

import logging
log = logging.getLogger(__name__)



class LaminographyCorrector(Processor):

    def __init__(self, initial_parameters=(30.0, 0.0), parameter_bounds=[(30, 40),(-10, 10)], 
                 parameter_tolerance=(0.01, 0.01), coarse_binning=None, final_binning = None, 
                 angle_binning = None, reduced_volume = None):
        """
        Initialize a LaminographyCorrector processor to optimize tilt and center-of-rotation for 
        laminography data.
        
        Parameters
        ----------
        initial_parameters : tuple of float, optional
            Initial guess for the geometry parameters (tilt_angle_deg, center_of_rotation_pix).
            Defaults to (30.0, 0.0).

        parameter_bounds : list of tuple of float, optional
            Bounds for the parameters [(tilt_min, tilt_max), (cor_min, cor_max)].
            Defaults to [(30, 40), (-10, 10)].

        parameter_tolerance : tuple of float, optional
            Convergence tolerance for optimisation of parameters, (tilt_tol, cor_tol).
            Defaults to (0.01, 0.01).

        coarse_binning : int, optional
            Initial binning factor applied to the input dataset for coarse optimisation.
            If None, a value based on dataset size is used.

        final_binning : int, optional
            Final binning factor applied for fine optimisation optimisation.
            If None, no binning is applied in the fine optimisation step.

        angle_binning : float, optional
            Subsampling factor for the angular dimension during optimisation.
            If None, automatically determined based on input dataset.

        reduced_volume : ImageGeometry, optional
            A reduced-resolution volume to be used for optimisation, e.g. obtained using 
            VolumeShrinker.
            If None, the full dataset is used.

            
        Example
        -------

        >>> processor = LaminographyCorrector(initial_parameters=(tilt, cor), 
            parameter_bounds=(tilt_bounds, cor_bounds), 
            parameter_tolerance=(tilt_tol, cor_tol))
        >>> processor.set_input(data)
        >>> data_corrected = processor.get_output()
        
        """
        kwargs = {
                    'initial_parameters'  : initial_parameters,
                    'parameter_bounds' : parameter_bounds,
                    'parameter_tolerance' : parameter_tolerance,
                    'reduced_volume' : reduced_volume,
                    'coarse_binning' : coarse_binning,
                    'final_binning' : final_binning,
                    'angle_binning' : angle_binning,
                    'evaluations' : []
                    }
        super(LaminographyCorrector, self).__init__(**kwargs)

    def check_input(self, dataset):
        if not isinstance(dataset, (AcquisitionData)):
            raise TypeError('Processor only supports AcquisitionData')
        
        if dataset.geometry.geom_type & AcquisitionType.CONE_FLEX:
            raise NotImplementedError("Processor not implemented for CONE_FLEX geometry.")
        
        if dataset.geometry.geom_type & AcquisitionType.CONE:
            raise NotImplementedError("LaminographyCorrector does not yet support CONE data")
        
        if not AcquisitionDimension.check_order_for_engine('astra', dataset.geometry):
            raise ValueError("LaminographyCorrector must be used with astra data order, try `data.reorder('astra')`")

        return True

    def update_geometry(self, ag, tilt_deg, cor_pix, 
                        tilt_direction_vector = np.array([1, 0, 0]), 
                        original_rotation_axis=np.array([0, 0, 1])):

        tilt_rad = np.deg2rad(tilt_deg)
        rotation_matrix = R.from_rotvec(tilt_rad * tilt_direction_vector)
        tilted_rotation_axis = rotation_matrix.apply(original_rotation_axis)

        ag.set_centre_of_rotation(offset=cor_pix, distance_units='pixels')
        ag.config.system.rotation_axis.direction = tilted_rotation_axis

        return ag
   
    def sobel_2d(self, arr):
        gx = sobel(arr, axis=0)
        gy = sobel(arr, axis=2)
        return np.sqrt(gx**2 + gy**2)

    def highpass_2d(self, arr, sigma=3.0):
        return arr - gaussian_filter(arr, sigma=sigma)
    
    def get_min(self, offsets, values, ind):
        #calculate quadratic from 3 points around ind  (-1,0,1)
        a = (values[ind+1] + values[ind-1] - 2*values[ind]) * 0.5
        b = a + values[ind] - values[ind-1]
        ind_centre = -b / (2*a)+ind

        ind0 = int(ind_centre)
        w1 = ind_centre - ind0
        return (1.0 - w1) * offsets[ind0] + w1 * offsets[ind0+1]
    
    def loss_from_residual(self, residual,
                       hp_sigma=3.0,
                       use_highpass=True,
                       use_sobel=True,
                       normalize_per_angle=False):
        
        r = residual.as_array()

        if use_highpass:
            r = self.highpass_2d(r, sigma=hp_sigma)
        if use_sobel:
            r = self.sobel_2d(r)

        return float(np.sum(r**2))
    
    def projection_reprojection(self, data, ig, ag, ag_ref, y_ref, tilt_deg, cor_pix):
        
        ag = self.update_geometry(ag, tilt_deg, cor_pix)
        recon = FBP(ig, ag)(data)
        recon.apply_circular_mask(0.9)

        ag_ref = self.update_geometry(ag_ref, tilt_deg, cor_pix)
        A = ProjectionOperator(ig, ag_ref)
        
        yhat = A.direct(recon)
        r = yhat - y_ref
        
        loss = self.loss_from_residual(r)
        
        return loss, recon
    
   
    def minimise_geometry(self, data, binning, p0, bounds, calculate_ftol):
        
        current_run_evaluations = []
        xtol = self.parameter_tolerance
        p0_binned = (p0[0], p0[1]/binning)
        bounds_binned = (bounds[0], (bounds[1][0]/binning, bounds[1][1]/binning))

        p0_scaled = np.array([p0_binned[0] / xtol[0],
                              p0_binned[1] / xtol[1]], dtype=float)
        
        bounds_scaled = [(bounds_binned[0][0] / xtol[0], bounds_binned[0][1] / xtol[0]),
                         (bounds_binned[1][0] / xtol[1],  bounds_binned[1][1] / xtol[1])]
        
        direc = np.diag(np.asarray(xtol) / np.min(xtol))
        
        print(f"Tilt bounds : ({bounds[0][0]:.3f}:{bounds[0][1]:.3f}), CoR bounds : ({bounds[1][0]:.3f}:{bounds[1][1]:.3f})")

        target = max(np.ceil(data.get_dimension_size('angle') / 10), 36)
        divider = np.floor(data.get_dimension_size('angle') / target)
        y_ref = Slicer(roi={'angle':(None, None, divider)})(data)

        ag = data.geometry.copy()
        ag_ref = Slicer(roi={'angle':(None, None, divider)})(ag)

        if self.reduced_volume is None:
            ig = ag.get_ImageGeometry()
        else:
            ig = Binner(roi={'horizontal_x':(None, None,binning), 'horizontal_y':(None, None,binning), 'vertical':(None, None,binning)})(self.reduced_volume)
        if calculate_ftol:
            loss_at_p0, _ = self.projection_reprojection(data, ig, ag, ag_ref, y_ref, p0_binned[0], p0_binned[1])
            ftol = self.ftol_from_bounds_and_xtol(loss_at_p0, 1, bounds_scaled)
        else:
            ftol = None
        
        def loss_function_wrapper(p):
            tilt = p[0] * xtol[0]
            cor  = p[1] * xtol[1]

            loss, recon = self.projection_reprojection(data, ig, ag, ag_ref, y_ref, tilt, cor)
            
            current_run_evaluations.append((tilt, cor * binning, loss))

            print(f"tilt: {tilt:.3f}, CoR: {cor*binning:.3f}, loss: {loss:.3e}")

            return loss
        
        if ftol is not None:
            res_scaled = minimize(loss_function_wrapper, p0_scaled,
                        method='Powell',
                        bounds=bounds_scaled,
                        options={'maxiter': 5, 'disp': True, 'ftol': ftol, 'xtol': 1.0, 'direc': direc})
        else:
           res_scaled = minimize(loss_function_wrapper, p0_scaled,
                        method='Powell',
                        bounds=bounds_scaled,
                        options={'maxiter': 5, 'disp': True, 'xtol': 1.0, 'direc': direc}) 
        
        res_real = res_scaled
        res_real.x = np.array([res_scaled.x[0] * xtol[0],
                           res_scaled.x[1] * xtol[1] * binning])
        
        self.evaluations.append({
            "p0": p0,
            "bounds": bounds,
            "binning": binning,
            "xtol": xtol,
            "result": res_real,
            "evaluations": current_run_evaluations
        })

        return res_real
    
    def ftol_from_bounds_and_xtol(self, loss_at_p0, xtol, bounds, min_abs_fatol=1e-6):

        xtol = np.asarray(xtol, dtype=float)
        ranges = np.array([b[1] - b[0] for b in bounds], dtype=float)
        tau = np.min(xtol / ranges)
        ftol = max(min_abs_fatol, tau * abs(loss_at_p0))
        return ftol
    
    def process(self, out=None):
        data = self.get_input()

        if self.coarse_binning is None:
            self.coarse_binning = min(int(np.ceil(data.geometry.config.panel.num_pixels[0] / 256)),5)
        binning = self.coarse_binning
        if self.angle_binning is None:
            self.angle_binning = np.ceil(data.get_dimension_size('angle')/(data.get_dimension_size('horizontal')*(np.pi/2)))
        roi = {
                'horizontal': (None, None, binning),
                'vertical': (None, None, binning),
                'angle': (None, None, self.angle_binning*binning)
            }

        data_binned = Binner(roi)(data)
        
        coarse_tolerance = (self.parameter_tolerance[0], self.parameter_tolerance[1])
        res = self.minimise_geometry(data_binned, binning=binning, 
                                                  p0=self.initial_parameters, 
                                                  bounds=self.parameter_bounds,
                                                  calculate_ftol=False)
        
        tilt_min = res.x[0]
        cor_min = res.x[1]
        print(f"Coarse scan optimised tilt = {tilt_min:.3f}, CoR = {cor_min:.3f}")

        if self.final_binning is None:
            binning = 1
        else:
            binning = self.final_binning
        roi = {
                'horizontal': (None, None, binning),
                'vertical': (None, None, binning),
                'angle': (None, None, self.angle_binning)
            }

        data_binned = Binner(roi)(data)

        search_factor = 2                # multiplier on parameter_tolerance
        min_search_range_tilt = 1.0      # deg
        min_search_range_cor  = 1.0      # pix

        half_width_tilt = max(search_factor * coarse_tolerance[0], min_search_range_tilt/2)
        fine_bounds_tilt = (tilt_min - half_width_tilt, tilt_min + half_width_tilt)

        half_width_cor = max(search_factor * coarse_tolerance[1], min_search_range_cor/2)
        fine_bounds_cor = (cor_min - half_width_cor, cor_min + half_width_cor)

        res = self.minimise_geometry(data_binned, binning=binning,
                                            p0=(tilt_min, cor_min), bounds=[fine_bounds_tilt, fine_bounds_cor], 
                                            calculate_ftol=True)
        tilt_min = res.x[0]
        cor_min = res.x[1]
        print(f"Fine scan optimised tilt = {tilt_min:.3f}, CoR ={cor_min:.3f}")
        
        if log.isEnabledFor(logging.DEBUG):
            self.plot_evaluations()

        new_geometry = data.geometry.copy()
        self.update_geometry(new_geometry, tilt_min, cor_min)

        if out is None:
            return AcquisitionData(array=data.as_array(), deep_copy=True, geometry=new_geometry)
        else:
            out.geometry = new_geometry
            return out
        
    def plot_evaluations(self):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
        for i in [0, 1]:
            eval = self.evaluations[i]
            tilts = [t[0] for t in eval['evaluations']]
            cors = [t[1] for t in eval['evaluations']]
            losses = [t[2] for t in eval['evaluations']]
            
            ax = axs[i]
            scatter = ax.scatter(tilts, cors, c=losses, cmap='viridis', s=100, edgecolors='k')
            fig.colorbar(scatter, label='Loss value', ax=ax)
            ax.set_xlabel('Tilt')
            ax.set_ylabel('Cor')
            ax.set_title('bounds = ({:.2f}:{:.2f}), ({:.2f}:{:.2f}), binning = {}, xtol = ({}, {}) \n result = ({:.3f}, {:.3f})'
                        .format(*eval['bounds'][0], *eval['bounds'][1], eval['binning'], *eval['xtol'], eval['result'].x[0], eval['result'].x[1]))
            ax.grid()
        plt.tight_layout()



