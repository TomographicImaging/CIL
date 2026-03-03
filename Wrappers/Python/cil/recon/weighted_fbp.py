import numpy as np

from cil.plugins.astra import FBP as FBP_ASTRA
import numba
from cil.framework.labels import AngleUnit, AcquisitionType
from cil.framework.acquisition_geometry import SystemConfiguration
import warnings
import logging

def calculate_angular_sampling_weights_cone(data, scan_type = 'full', max_gap=None, wedge_behaviour='forward/back'):
    '''
    Calculates angular sampling weights for cone beam data, based on the spacing of the projection angles.
    Wraps around the angular domain to calculate the spacing between the first and last projection angles, unless this is determined to be a wedge gap based on the max_gap parameter.

    Parameters
    ----------
    scan_type: {'full', 'half'}, default: 'full'
        This defines the ideal minimum angular range needed for the scan.
        If 'full' this is 360 degrees.
        If 'half' this is 180 degrees plus the cone angle. 
        
        The spacing between the last and first projection angles is computed 
        by wrapping around the angular domain, unless the spacing is determined to be a wedge gap,
        depending on the max_gap parameter.
    
    max_gap: This is used to find missing wedges in the data: max_gap is the maximum spacing between angles which 
        isn't classified as a wedge. If a gap between two angles is found to be greater than the max_gap, it is assigned 
        a value based on the selected 'wedge_behaviour'. 
        Default behaviour:
            If max_gap is None:
                This is set to double the mean of all gaps. 

    wedge_behaviour: {'forward/back', 'max_gap'}
        If 'forward/back', the angular spacing assigned to the final angle before a wedge and the first angle after a wedge are the backward or forward 
        gaps respectively.

        If 'max_gap', the angular spacing assigned to the final angle before a wedge and the first angle after a wedge is determined by max_gap.
        If 'max_gap' has been set by the user then these are set to 'max_gap'/2

    Returns
    -------
    weights: np.ndarray
        An array of weights with the same length as the number of projections in the data, which can be used to weight the projections for FBP reconstruction.
        These require normalisation before use in FBP, which can be done with the normalise_weights_for_FBP function.

    
    '''

    if data.geometry.geom_type & AcquisitionType.CONE_FLEX:
        raise NotImplementedError("calculate_angular_sampling_weights_cone does not currently support CONE_FLEX geometries")
    if not data.geometry.geom_type & AcquisitionType.CONE:
        raise ValueError("calculate_angular_sampling_weights_cone only supports cone beam data")

    if scan_type == 'full':
        angular_domain = 360
    elif scan_type == 'half':
        # When implemented this will calculate 180 + cone angle
        raise NotImplementedError("Half scan type is currently not implemented.")
    else:
        raise ValueError("Currently only 'full' scan_type is supported.")
    


    return _calculate_angular_sampling_weights(data, angular_domain, max_gap, wedge_behaviour)

    

def calculate_angular_sampling_weights_parallel(data, max_gap=None, wedge_behaviour='forward/back'):
    '''
    Calculates angular sampling weights for parallel beam data, based on the spacing of the projection angles.
    Wraps around the angular domain to calculate the spacing between the first and last projection angles, unless this is determined to be a wedge gap based on the max_gap parameter.
   
    Parameters
    ----------
    max_gap: This is used to find missing wedges in the data: max_gap is the maximum spacing between angles which 
        isn't classified as a wedge. If a gap between two angles is found to be greater than the max_gap, it is assigned 
        a value based on the selected 'wedge_behaviour'.
        Default behaviour:
            If max_gap is None: 
                This is set to double the mean of all gaps. 

    wedge_behaviour: {'forward/back', 'max_gap'}
        If 'forward/back', the angular spacing assigned to the final angle before a wedge and the first angle after a wedge are the backward or forward 
        gaps respectively.

        If 'max_gap', the angular spacing assigned to the final angle before a wedge and the first angle after a wedge is determined by max_gap.
        If 'max_gap' has been set by the user then these are set to 'max_gap'/2
    
    Returns
    -------
    weights: np.ndarray
        An array of weights with the same length as the number of projections in the data, which can be used to weight the projections for FBP reconstruction.
        These require normalisation before use in FBP, which can be done with the normalise_weights_for_FBP function.
    '''
    
    
    if not data.geometry.geom_type & AcquisitionType.PARALLEL:
        raise ValueError("calculate_angular_sampling_weights_parallel only supports parallel beam data")

    if data.geometry.config.system.system_description == SystemConfiguration.SYSTEM_ADVANCED:
        tilt = data.geometry.get_centre_of_rotation()['angle']
        tilt_value = tilt[0]
        tilt_unit = tilt[1]
        #TODO: figure out what tolerance should be used here and make unit tests
        percentage_tolerance = 0.05
        if tilt_unit == AngleUnit.DEGREE:
            max_tilt = 90 * percentage_tolerance
        elif tilt_unit == AngleUnit.RADIAN:
            max_tilt = np.pi/2 * percentage_tolerance
        
        max_tilt = max_tilt * (percentage_tolerance)
        if tilt_value > max_tilt:
            raise NotImplementedError("calculate_angular_sampling_weights_parallel does not currently support tilted geometries with tilt greater than {max_tilt:.2f} {tilt_unit}".format(max_tilt=max_tilt, tilt_unit=tilt_unit))
    return _calculate_angular_sampling_weights(data, 180, max_gap, wedge_behaviour)
  
    
def _calculate_angular_sampling_weights(data, angular_domain, max_gap=None, wedge_behaviour='forward/back'):  #, wrap_angles=True
    '''
    angular_domain: (180,360, or 180 plus cone angle)
        The ideal minimum angular range needed for the scan. This is not the angular range covered, its the angular range the recon needs
        default assumes 180 for parallel beam and 360 for cone beam
        If you had a laminography parallel beam for example you would need angular_domain=360
        The spacing between the last and first projection angles is computed 
        by wrapping around the angular domain, unless the spacing is determined to be a wedge gap,
        depending on the max_gap parameter.
    
    max_gap: This is used to find missing wedges in the data: max_gap is the maximum spacing between angles which 
        isn't classified as a wedge. If a gap between two angles is found to be greater than the max_gap, it is assigned 
        a value based on the selected 'wedge_behaviour'. 
        Default behaviour:
            If max_gap is None: # TODO: could be 1 degree - what does this mean for a 4k detector?
                This is set to double the mean of all gaps.

    wedge_behaviour: {'forward/back', 'max_gap'}
        If 'forward/back', the angular spacing assigned to the final angle before a wedge and the first angle after a wedge are the backward or forward 
        gaps respectively.

        If 'max_gap', the angular spacing assigned to the final angle before a wedge and the first angle after a wedge is determined by max_gap.
        If 'max_gap' has been set by the user then these are set to 'max_gap'/2

    Returns
    -------
    weights: np.ndarray
        An array of weights with the same length as the number of projections in the data, which can be used to weight the projections for FBP reconstruction.
        These require normalisation before use in FBP, which can be done with the normalise_weights_for_FBP function.
    '''
    angles = data.geometry.angles.copy()

    if max_gap is None:
        if wedge_behaviour == 'max_gap':
            raise ValueError("If wedge_behaviour is set to 'max_gap', max_gap must be set.")
    if wedge_behaviour not in ['forward/back', 'max_gap']:
        raise ValueError("wedge_behaviour must be either 'forward/back' or 'max_gap'")


            
    num_proj = data.geometry.num_projections

    # Create array of angle indices in order of angle values
    sorted_indices = np.argsort(angles % angular_domain)
    ordered_mod_angles = angles[sorted_indices] % angular_domain

    num_proj = len(ordered_mod_angles)
    weights = np.zeros(num_proj)
    

    gaps=[]
    for i in range(num_proj):
        prev_idx = (i - 1) % num_proj
        gap = (ordered_mod_angles[i]-ordered_mod_angles[prev_idx]) % angular_domain
        gaps.append(gap)

    gaps = np.asarray(gaps)

    mean_gap = np.mean(gaps)
    if np.allclose(gaps, mean_gap):
        return np.array([mean_gap]*num_proj)  

    max_angular_span = max_gap

    # If the user has not set the max_gap:

    if max_angular_span is None:
        gaps_no_zeros = gaps[~np.isclose(gaps, 0, atol=1e-4)]
        # We need to identify whether there are any missing wedges in the data.
        # we identify any gap greater than mean_gap *2 to be a wedge

        max_angular_span = np.mean(gaps_no_zeros)*2 # duplicate angles shouldn't go into the mean
        # calculate mean of all gaps that aren't wedge gaps: 

        warnings.warn("Because max_gap was None, it has been set to double the mean of all gaps between angles: " \
        "{:.4f} {angular_unit}".format(max_angular_span, angular_unit=data.geometry.config.angles.angle_unit), UserWarning, stacklevel=2)      

    else:
        default_gap = max_angular_span

    num_remaining_proj_at_current_angle = 1
    total_num_proj_at_current_angle=1
    total_duplicates = 0
    num_wedges=0

    for i in range(num_proj):
        prev_idx = (i - 1 - (total_num_proj_at_current_angle-num_remaining_proj_at_current_angle)) % num_proj
        next_idx = (i + num_remaining_proj_at_current_angle) % num_proj
        current_angle = ordered_mod_angles[i]
        prev_angle = ordered_mod_angles[prev_idx]
        next_angle = ordered_mod_angles[next_idx]

        if total_num_proj_at_current_angle == 1: # This means we haven't checked if there are any projections at this same angle:
            if np.isclose(next_angle, current_angle):
                while np.isclose(next_angle, current_angle):
                    next_idx = (next_idx + 1) % num_proj
                    next_angle = ordered_mod_angles[next_idx]
                    num_remaining_proj_at_current_angle += 1
                total_num_proj_at_current_angle = num_remaining_proj_at_current_angle
                total_duplicates+=total_num_proj_at_current_angle-1

        angle_coverage = ((next_angle - prev_angle) % angular_domain) / 2.0
        if angle_coverage > max_angular_span:
            num_wedges+=1
            if wedge_behaviour == 'forward/back':
                forward = (next_angle - current_angle) % angular_domain
                backward = (current_angle - prev_angle) % angular_domain
                if np.isclose(forward, backward):
                    angle_coverage = max_angular_span
                else:
                    angle_coverage = min(forward, backward)
            else:
                angle_coverage = default_gap

        weights[i] = angle_coverage/ total_num_proj_at_current_angle

        if total_num_proj_at_current_angle > 1:
            # if we have additional projections at the same angle that we already knew of,
            # reduce the counter of remaining projections at this angle for next loop:
            num_remaining_proj_at_current_angle-=1
            if num_remaining_proj_at_current_angle == 0:
                # This means we have processed all of the duplicates in this block. Reset counters for duplicates:
                total_num_proj_at_current_angle = 1
                num_remaining_proj_at_current_angle = 1

    if num_wedges > 0:
        logging.info(f"{num_wedges} missing wedges were identified in the data based on the max_gap value of {max_angular_span:.4f} {data.geometry.config.angles.angle_unit}. If this is not expected, consider adjusting the max_gap value.")

    # Reorder weights back to original projection order
    original_order_weights = np.zeros(num_proj)
    for i, idx in enumerate(sorted_indices):
        original_order_weights[idx] = weights[i]

    return original_order_weights


def normalise_weights_for_FBP(weights):
    '''
    Normalises weights for use in FBP or FDK reconstruction, by ensuring
    the sum of the weights is equal to the number of projections in the data.

    The weights are divided by sum of the weights and multiplied by the number of weights.

    Parameters
    ----------
    weights: np.ndarray
        An array of weights with the same length as the number of projections in the data, which
        can be used to weight the projections for FBP reconstruction.
        These are typically the output of the calculate_angular_sampling_weights_cone or calculate_angular_sampling_weights_parallel functions,
        or can be custom weights defined by the user.

    Returns
    -------
    normalised_weights: np.ndarray
        An array of normalised weights with the same length as the number of projections in the data.
        These can be used to weight the projections for FBP reconstruction.
    '''

    normalised_weights = weights.copy()
    normalised_weights = normalised_weights / np.sum(normalised_weights) * len(normalised_weights)
    return normalised_weights




def run_weighted_fbp(data, weights, accelerated=True):
    if 'vertical' in data.dimension_labels:
        v_size = data.get_dimension_size('vertical')
    else:
        v_size = 1

    if 'horizontal' in data.dimension_labels:
        h_size = data.get_dimension_size('horizontal')
    else:
        h_size = 1

    proj_size = v_size * h_size
    num_proj = int(data.array.size / proj_size)

    data_weighted = data.copy()  # Doesn't alter the original AcquisitionData
    data_weighted_array = data_weighted.array

    data_weighted_reshaped = data_weighted_array.reshape(num_proj, proj_size)
    

    if accelerated:
        numba_loop(weights, num_proj, proj_size, data_weighted_array)
    else:
        weights = np.asarray(weights)[:, np.newaxis]  # shape: (num_proj, 1)
        np.multiply(data_weighted_reshaped, weights, out=data_weighted_reshaped)

    data_weighted.array = data_weighted_array
    print(data_weighted)
    data_weighted.reorder('astra')
    recon = FBP_ASTRA(data_weighted.geometry.get_ImageGeometry(), data_weighted.geometry)(data_weighted)
    return recon


@numba.njit(parallel=True)
def numba_loop(weights, num_proj, proj_size, out):
    out_flat = out.ravel()
    for i in numba.prange(num_proj):
        for ij in range(proj_size):
            out_flat[i*proj_size+ij] *= weights[i]



# @numba.njit(parallel=True)
# def numba_loop(flux, target, num_proj, proj_size, out):
#     out_flat = out.ravel()
#     flux_flat = flux.ravel()
#     for i in numba.prange(num_proj):
#         for ij in range(proj_size):
#             out_flat[i*proj_size+ij] *= (target/flux_flat[i])
