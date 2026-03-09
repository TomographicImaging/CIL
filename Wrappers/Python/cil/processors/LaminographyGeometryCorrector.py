#  Copyright 2026 United Kingdom Research and Innovation
#  Copyright 2026 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter, sobel
import matplotlib.pyplot as plt
import numpy as np
import importlib

from cil.framework import Processor
from cil.processors import Binner, Slicer
from cil.framework import AcquisitionData
from cil.framework.labels import AcquisitionType, AcquisitionDimension

import logging
log = logging.getLogger(__name__)



class LaminographyGeometryCorrector(Processor):
    """
        LaminographyGeometryCorrector processor to fit the geometry of a 
        parallel beam laminography dataset to find tilt and center-of-rotation.
        
        Parameters
        ----------
        parameter_bounds : list of tuple of float, optional
            Bounds for the parameters [(tilt_min_deg, tilt_max_deg), (CoR_min_pix, CoR_max_pix)].
            Defaults to [(-10, 10), (-20, 20)].

        parameter_tolerance : tuple of float, optional
            Convergence tolerance for optimisation of parameters, (tilt_tol_deg, CoR_tol_pix).
            Defaults to (0.01, 0.01).

        coarse_binning : int, optional
            Initial binning factor applied to the input dataset for coarse optimisation.
            If None, a value based on dataset size is used.

        final_binning : int, optional
            Final binning factor applied for fine optimisation.
            If None, no binning is applied in the fine optimisation step.

        angle_subsampling : float, optional
            Subsampling factor for the angle dimension during optimisation.
            If None, automatically determined based on input dataset.

        image_geometry : ImageGeometry, optional
            Pass a reduced volume ImageGeometry to be used for fitting.
            If None, the full dataset is used.

        backend : {'astra'}, optional
            The backend to use for the reconstruction. Currently only 'astra'
            is supported.

            
        Example
        -------

        >>> processor = LaminographyGeometryCorrector(parameter_bounds=(tilt_bounds, CoR_bounds), 
            parameter_tolerance=(tilt_tol, CoR_tol))
        >>> processor.set_input(data)
        >>> data_corrected = processor.get_output()
        
        """
    
    _supported_backends = ['astra']

    def __init__(self, parameter_bounds=[(-10, 10),(-20, 20)], parameter_tolerance=(0.01, 0.01), 
                 coarse_binning=None, final_binning = None, angle_subsampling = None, image_geometry = None, backend='astra'):
        

        FBP, ProjectionOperator = self._configure_FBP(backend)
        kwargs = {
                    'initial_parameters'  : None,
                    'parameter_bounds' : parameter_bounds,
                    'parameter_tolerance' : parameter_tolerance,
                    'image_geometry' : image_geometry,
                    'coarse_binning' : coarse_binning,
                    'final_binning' : final_binning,
                    'angle_subsampling' : angle_subsampling,
                    'backend' : backend,
                    'FBP' : FBP,
                    'ProjectionOperator' : ProjectionOperator,
                    'evaluations' : []
                    }
        super(LaminographyGeometryCorrector, self).__init__(**kwargs)

    def check_input(self, dataset):
        if not isinstance(dataset, (AcquisitionData)):
            raise TypeError('Processor only supports AcquisitionData')
        
        if dataset.geometry.geom_type & AcquisitionType.CONE_FLEX:
            raise NotImplementedError("Processor not implemented for CONE_FLEX geometry.")
        
        if dataset.geometry.geom_type & AcquisitionType.CONE:
            raise NotImplementedError("LaminographyGeometryCorrector does not yet support CONE data")
        
        if not AcquisitionDimension.check_order_for_engine('astra', dataset.geometry):
            raise ValueError("LaminographyGeometryCorrector must be used with astra data order, try `data.reorder('astra')`")
        
        if not dataset.geometry.dimension & AcquisitionType.DIM3:
            raise ValueError("LaminographyGeometryCorrector must be used 3D data")

        return True

    def _get_initial_parameters(self):
        """
        Get the current tilt and centre of rotation from the geometry
        """
        # get initial parameters from geometry
        dataset = self.get_input()
        U = dataset.geometry.config.system.rotation_axis.direction
        V = dataset.geometry.config.system.detector.direction_y
        c = np.cross(U, V)
        d = np.dot(U, V)
        c_norm = np.linalg.norm(c)
        tilt_deg = np.rad2deg(np.arctan2(c_norm, d))
        CoR_pix = dataset.geometry.get_centre_of_rotation('pixels')['offset'][0]
        self.initial_parameters = (tilt_deg, CoR_pix)

    def _update_geometry(self, ag, tilt_deg, cor_pix, 
                        tilt_direction_vector = np.array([1, 0, 0]), 
                        original_rotation_axis=np.array([0, 0, 1])):
        """
        Update the rotation matrix direction and centre of rotation from a tilt
        in degrees and centre of rotation offset in pixels     
        """    
        tilt_rad = np.deg2rad(tilt_deg)
        rotation_matrix = R.from_rotvec(tilt_rad * tilt_direction_vector)
        tilted_rotation_axis = rotation_matrix.apply(original_rotation_axis)

        ag.set_centre_of_rotation(offset=cor_pix, distance_units='pixels')
        ag.config.system.rotation_axis.direction = tilted_rotation_axis

        return ag
       
    def _projection_reprojection(self, data, recon, ig, ag, ag_downsampled, data_downsampled, residual, tilt_deg, cor_pix):
        """
        Reconstruct the data then re-project and calculate the residual. Then
        filter the residual and calculate the L2Norm loss.

        Parameters
        ----------
        data: AcquisitionData
            The full size dataset
        recon: ImageData
            Pre-allocated reconstruction volume
        ig: ImageGeometry
            Reconstruction volume geometry
        ag: AcquisitionGeometry
            Copy of full data geometry
        ag_downsampled: AcquisitionGeometry
            Geometry of the downsampled dataset used for reprojection
        data_downsampled: AcquisitionData
            Downsampled dataset used for reprojection
        residual: AcquisitionData
            Pre-allocated residual, difference between data and reprojection
        tilt_deg: float
            Latest tilt angle in degrees
        cor_pix: float
            Latest centre of rotation offset in pixels

        """
        
        # update the geometry with latest values and get reconstruction
        ag = self._update_geometry(ag, tilt_deg, cor_pix)
        FBP = self.FBP(ig, ag)
        FBP.set_input(data)
        FBP.get_output(out=recon)
        recon.apply_circular_mask(0.9)

        # update the downsampled data geometry and get forward projection
        ag_downsampled = self._update_geometry(ag_downsampled, tilt_deg, cor_pix)
        A = self.ProjectionOperator(ig, ag_downsampled)
        A.direct(recon, out=residual)
        # subtract the downsampled reference data
        residual.subtract(data_downsampled, out=residual)
        
        # apply Gaussian and Sobel filter - note the axes are hard coded here for astra, this would need to be updated for tigre 
        residual.subtract(gaussian_filter(residual.as_array(), sigma=3.0, axes=(0,2)), out=residual)
        np.sqrt((sobel(residual.array, axis=0))**2 + (sobel(residual.array, axis=2))**2, out=residual.array)
        
        loss = float(np.sum(residual**2))

        return loss
    
    def _minimise_geometry(self, data, binning, p0, bounds):
        """
        Setup and run the scipy Powell minimize method

        Parameters
        ----------
        data: AcquistionData
            Full dataset
        binning: int
            Current detector binning value
        p0: list or tuple
            Initial start parameters for search (tilt_deg, cor_pix)
        bounds: list of tuple of floats
            Bounds for the parameters [(tilt_min_deg, tilt_max_deg), (CoR_min_pix, CoR_max_pix)]
        """
        
        current_run_evaluations = []
        xtol = self.parameter_tolerance

        # scale the start values and bounds by the binning
        p0_binned = (p0[0], p0[1]/binning)
        bounds_binned = (bounds[0], (bounds[1][0]/binning, bounds[1][1]/binning))

        # scale the start values and bounds so xtol can be 1
        p0_scaled = np.array([p0_binned[0] / xtol[0],
                              p0_binned[1] / xtol[1]], dtype=float)
        
        bounds_scaled = [(bounds_binned[0][0] / xtol[0], bounds_binned[0][1] / xtol[0]),
                         (bounds_binned[1][0] / xtol[1],  bounds_binned[1][1] / xtol[1])]
        
        direc = np.diag(np.asarray(xtol) / np.min(xtol))
        
        # get y_ref: a subset of the real data to compare with the reprojections
        target = max(np.ceil(data.get_dimension_size('angle') / 10), 36)
        divider = np.floor(data.get_dimension_size('angle') / target)
        data_downsampled = Slicer(roi={'angle':(None, None, divider)})(data)

        # also get a matching reference geometry
        ag = data.geometry.copy()
        ag_downsampled = Slicer(roi={'angle':(None, None, divider)})(ag)

        if self.image_geometry is None:
            ig = ag.get_ImageGeometry()
        else:
            ig = Binner(roi={'horizontal_x':(None, None,binning), 'horizontal_y':(None, None,binning), 'vertical':(None, None,binning)})(self.image_geometry)
        
        # pre-allocate reconstruction volume and residual array
        recon = ig.allocate(0)
        residual = ag_downsampled.allocate()

        def loss_function_wrapper(p):
            """
            Function wrapper for the loss function, self._projection_reprojection
            to be called by scipy.minimize. Rescales tilt and cor by xtol so a 
            single tolerance can be used for each parameter.
            """
            tilt = p[0] * xtol[0]
            cor  = p[1] * xtol[1]
            loss = self._projection_reprojection(data, recon, ig, ag, ag_downsampled, data_downsampled, residual, tilt, cor)
            
            current_run_evaluations.append((tilt, cor * binning, loss))

            print(f"tilt: {tilt:.3f}, CoR: {cor*binning:.3f}, loss: {loss:.3e}")

            return loss
        
        # call minimize
        res_scaled = minimize(loss_function_wrapper, p0_scaled,
                    method='Powell',
                    bounds=bounds_scaled,
                    options={'maxiter': 5, 'disp': True, 'xtol': 1.0, 'direc': direc}) 
        
        # re-scale the results
        res_real = res_scaled
        res_real.x = np.array([res_scaled.x[0] * xtol[0],
                           res_scaled.x[1] * xtol[1] * binning])
        
        # save information about the minimisation
        self.evaluations.append({
            "p0": p0,
            "bounds": bounds,
            "binning": binning,
            "xtol": xtol,
            "result": res_real,
            "evaluations": current_run_evaluations
        })

        return res_real
    
    def process(self, out=None):

        data = self.get_input()
        self._get_initial_parameters()

        # apply coarse binning to the data
        if self.coarse_binning is None:
            # if no coarse binning provided, get a default binning based on the size of the panel
            self.coarse_binning = min(int(np.ceil(data.geometry.config.panel.num_pixels[0] / 256)),5)
        binning = self.coarse_binning
        roi = {
                'horizontal': (None, None, binning),
                'vertical': (None, None, binning)
            }
        data_binned = Binner(roi)(data)

        # sub-sample the angles
        if self.angle_subsampling is None:
            # if no sub-sampling value is provided, get a default subsampling based on the Nyquist criteria
            self.angle_subsampling = np.ceil(data.get_dimension_size('angle')/(data.get_dimension_size('horizontal')*(np.pi/2)))
        roi={'angle':(None, None, self.angle_subsampling*binning)}
        data_binned = Slicer(roi)(data_binned)
        
        # run coarse minimisation
        coarse_tolerance = (self.parameter_tolerance[0], self.parameter_tolerance[1])
        res = self._minimise_geometry(data_binned, binning=binning, 
                                                  p0=self.initial_parameters, 
                                                  bounds=self.parameter_bounds)
        
        tilt_min = res.x[0]
        cor_min = res.x[1]
        print(f"Coarse scan optimised tilt = {tilt_min:.3f}, CoR = {cor_min:.3f}")

        # apply final binning
        if self.final_binning is None:
            binning = 1
        else:
            binning = self.final_binning
        roi = {
                'horizontal': (None, None, binning),
                'vertical': (None, None, binning),
                'angle': (None, None, self.angle_subsampling)
            }

        data_binned = Binner(roi)(data)

        # calculate new search ranges based on coarse minimisation results
        search_factor = 2                # multiplier on parameter_tolerance
        min_search_range_tilt = 1.0
        min_search_range_cor  = 1.0

        half_width_tilt = max(search_factor * coarse_tolerance[0], min_search_range_tilt/2)
        fine_bounds_tilt = (tilt_min - half_width_tilt, tilt_min + half_width_tilt)

        half_width_cor = max(search_factor * coarse_tolerance[1], min_search_range_cor/2)
        fine_bounds_cor = (cor_min - half_width_cor, cor_min + half_width_cor)

        # run fine minimisation
        res = self._minimise_geometry(data_binned, binning=binning,
                                            p0=(tilt_min, cor_min), 
                                            bounds=[fine_bounds_tilt, fine_bounds_cor])
        tilt_min = res.x[0]
        cor_min = res.x[1]
        print(f"Fine scan optimised tilt = {tilt_min:.3f}, CoR ={cor_min:.3f}")
        
        if log.isEnabledFor(logging.DEBUG):
            self.plot_evaluations()

        new_geometry = data.geometry.copy()
        self._update_geometry(new_geometry, tilt_min, cor_min)

        if out is None:
            return AcquisitionData(array=data.as_array(), deep_copy=True, geometry=new_geometry)
        else:
            out.geometry = new_geometry
            return out
        
    def plot_evaluations(self):
        """
        Plot results from the minimisation. Plots the loss function value as a 
        function of tilt and centre of rotation offset position.
        """
        num_evals = len(self.evaluations)
        if num_evals > 0:
            fig, axs = plt.subplots(nrows=1, ncols=num_evals, figsize=(14, 6))
            
            for i in np.arange(num_evals):
                eval = self.evaluations[i]
                tilts = [t[0] for t in eval['evaluations']]
                cors = [t[1] for t in eval['evaluations']]
                losses = [t[2] for t in eval['evaluations']]
                
                ax = axs[i]
                scatter = ax.scatter(tilts, cors, c=losses, cmap='viridis_r', s=100, edgecolors='k')
                fig.colorbar(scatter, label='Loss value', ax=ax)
                ax.set_xlabel('Tilt')
                ax.set_ylabel('Cor')
                ax.set_title('bounds = ({:.2f}:{:.2f}), ({:.2f}:{:.2f}), binning = {}, xtol = ({}, {}) \n result = ({:.3f}, {:.3f})'
                            .format(*eval['bounds'][0], *eval['bounds'][1], eval['binning'], *eval['xtol'], eval['result'].x[0], eval['result'].x[1]))
                ax.grid()
            plt.tight_layout()
        else:
            raise ValueError("No evaluation available to plot. Run processor.process() first.")

    def _configure_FBP(self, backend='astra'):
        """
        Configures the recon and projection operator for the right engine.
        """
        if backend not in self._supported_backends:
            raise ValueError("Backend unsupported. Supported backends: {}".format(self._supported_backends))

        module = importlib.import_module(f'cil.plugins.{backend}')

        return module.FBP, module.ProjectionOperator



