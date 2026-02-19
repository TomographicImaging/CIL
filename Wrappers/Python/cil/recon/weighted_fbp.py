import numpy as np

from cil.plugins.astra import FBP as FBP_ASTRA
import numba
from cil.framework.labels import AngleUnit, AcquisitionType



def run_weighted_fbp(data, weights):
    # Apply weights to the data
    data_weighted = data.copy()
    for angle_idx in range(data.geometry.angles.size):
        try:
            data_weighted.array[angle_idx, :, :] *= weights[angle_idx]
        except:
            data_weighted.array[angle_idx, :] *= weights[angle_idx]

    # Reorder and reconstruct with weighted data
    data_weighted.reorder('astra')
    recon = FBP_ASTRA(data_weighted.geometry.get_ImageGeometry(), data_weighted.geometry)(data_weighted)
    return recon


def get_weights_for_FBP(data):
    angles = data.geometry.angles.copy()

    

    if data.geometry.config.angles.angle_unit == AngleUnit.DEGREE:
        half_angle = 180.0
    else:
        half_angle = np.pi

    if data.geometry.geom_type == AcquisitionType.CONE:
        angular_domain = half_angle*2.0
    elif data.geometry.geom_type & AcquisitionType.CONE_FLEX:
        raise NotImplementedError("get_weights_for_FBP does not currently support CONE_FLEX geometries")
    elif data.geometry.geom_type == AcquisitionType.PARALLEL:
        angular_domain = half_angle

    num_proj = data.geometry.num_projections

    return get_weights(angles, angular_domain, num_proj)
  
    
def calculate_angular_sampling_weights(data, angular_domain=None, max_gap=None, wedge_behaviour='forward/back'):  #, wrap_angles=True
    '''
    angular_domain: The ideal minimum angular range needed for the scan.
        default assumes 180 for parallel beam and 360 for cone beam
        If you had a laminography parallel beam for example you would need 360
        The spacing between the last and first projection angles is computed 
        by wrapping around the angular domain. This may or may not be determined to be a wedge gap,
        depending on the max_gap parameter.
    
    max_gap: This is used to find missing wedges in the data: max_gap is the maximum spacing between angles which 
        isn't classified as a wedge. If a gap between two angles is found to be greater than the max_gap, it is assigned 
        a value based on the selected 'wedge_behaviour'.
        Default behaviour:
            If max_gap is None:
                This is set to double the mean of all gaps.

    wedge_behaviour: {'forward/back', 'max_gap'}
        If 'forward/back', the angular spacing assigned to the final angle before a wedge and the first angle after a wedge are the doubled backward or forward 
        gaps respectively.
        If 'max_gap', the angular spacing assigned to the final angle before a wedge and the first angle after a wedge is determined by max_gap.
        If 'max_gap' has been set by the user then these are set to 'max_gap'
        However, if 'max_gap' is None:
            The replacement value is set to the mean of all gaps less than double the mean of all gaps.
            e.g. if you had limited angle data which spanned 70 degrees, with 10 degree gaps in this range and angular_domain=180:
            [0,10,20,30,40,50,60]
            gaps=[120,10,10,10,10,10,10]
            mean gap = 25.5
            max_gap = 51
            The replacement gap would be set to the mean of the gaps < 51, therefore 10
            so the weight for 0 would be 10, and for 60 would be 10 instead of the weight for 0 being (0-60)%180=120 
    '''
    angles = data.geometry.angles.copy()

    if data.geometry.geom_type & AcquisitionType.CONE_FLEX:
            raise NotImplementedError("X does not currently support CONE_FLEX geometries")

    if angular_domain is None:
        if data.geometry.config.angles.angle_unit == AngleUnit.DEGREE:
            half_angle = 180.0
        else:
            half_angle = np.pi

        angular_domain = half_angle

        if data.geometry.geom_type == AcquisitionType.CONE:
            angular_domain *=2.0
            
    num_proj = data.geometry.num_projections

    return get_weights(angles, angular_domain, num_proj, max_gap, wedge_behaviour)


def get_weights(angles, angular_domain, num_proj, max_gap, wedge_behaviour):
    # Create array of angle indices in order of angle values
    sorted_indices = np.argsort(angles % angular_domain)
    ordered_mod_angles = angles[sorted_indices] % angular_domain

    num_angles = len(ordered_mod_angles)
    weights = np.zeros(num_angles)
    

    # create tolerance for two angles being equal:
    gaps=[]
    for i in range(num_angles):
        prev_idx = (i - 1) % num_angles
        gap = (ordered_mod_angles[i]-ordered_mod_angles[prev_idx]) % angular_domain
        gaps.append(gap)

    gaps = np.asarray(gaps)
    

    max_angular_span = max_gap

    # If the user has not set the max_gap:

    if max_angular_span is None:
        gaps_no_zeros = gaps[~np.isclose(gaps, 0)]
        # We need to identify whether there are any missing wedges in the data.
        # we identify any gap greater than mean_gap *2 to be a wedge

        max_angular_span = np.mean(gaps_no_zeros)*2 # duplicate angles shouldn't go into the mean
        # calculate mean of all gaps that aren't wedge gaps: 
        default_gap = np.mean(gaps_no_zeros[gaps_no_zeros< max_angular_span])
    else:
        default_gap = max_angular_span


    print("Max angular span: ", max_angular_span)
    # TODO: decide what should be
    angular_tolerance = 0.01 * max_angular_span / 2


    # TODO: reinstate after testing:
    # check if all gaps are equal, if so return weights of 1:
    # if np.allclose(gaps, mean_gap, atol=angular_tolerance):
    #     return np.ones_like(angles)   

    num_proj_at_same_angle = 1
    total_duplicates = 0

    for i in range(num_angles):

        # skips if current angle same as prev angle as has already been assigned a weight:
        if num_proj_at_same_angle > 1:
            num_proj_at_same_angle -= 1
            continue

        prev_idx = (i - 1) % num_angles
        next_idx = (i + 1) % num_angles

        prev_angle = ordered_mod_angles[prev_idx]
        next_angle = ordered_mod_angles[next_idx]
        current_angle = ordered_mod_angles[i]
        # print("Current angle, ", current_angle)

       
        if np.isclose(next_angle, current_angle, atol=angular_tolerance):
            print(next_angle, current_angle)
            while np.isclose(next_angle, current_angle, atol=angular_tolerance):
                next_idx = (next_idx + 1) % num_angles
                next_angle = ordered_mod_angles[next_idx]
                num_proj_at_same_angle += 1
            print("Num proj at same angle: ", num_proj_at_same_angle)
            for j in range(num_proj_at_same_angle):
                current_angle = ordered_mod_angles[i + j]
                print("Next angle: ", "prev angle: ")
                angle_coverage = ((next_angle - prev_angle) % angular_domain) / 2.0
                print("Angle coverage: ", angle_coverage)
                if angle_coverage > max_angular_span:
                    if wedge_behaviour == 'forward/back':
                        forward = (next_angle - current_angle) % angular_domain
                        backward = (current_angle - prev_angle) % angular_domain
                        print("Forward: ", forward, "Backward: ", backward)
                        if np.isclose(forward, backward, atol=angular_tolerance):
                            angle_coverage = max_angular_span
                        else:
                            angle_coverage = min(forward, backward)
                else:
                    angle_coverage = default_gap
                print("Current angle ", ordered_mod_angles[i+j])
                print("Weight: ", angle_coverage/ num_proj_at_same_angle)
                weights[i+j] = angle_coverage/ num_proj_at_same_angle
            total_duplicates+=num_proj_at_same_angle-1

        else:
            angle_coverage = ((next_angle - prev_angle) % angular_domain) / 2.0
            print("Angle coverage: ", angle_coverage)
            if angle_coverage > max_angular_span:
                if wedge_behaviour == 'forward/back':
                    forward = (next_angle - current_angle) % angular_domain
                    backward = (current_angle - prev_angle) % angular_domain
                    print("Forward: ", forward, "Backward: ", backward)
                    if np.isclose(forward, backward, atol=angular_tolerance):
                        angle_coverage = max_angular_span
                    else:
                        angle_coverage = min(forward, backward)
            else:
                angle_coverage = default_gap
            # print("Weight: ", angle_coverage)
            weights[i] = angle_coverage

    # Reorder weights back to original projection order
    original_order_weights = np.zeros(num_angles)
    for i, idx in enumerate(sorted_indices):
        original_order_weights[idx] = weights[i]

    print("Before normalization:")
    print("Total of weights: ", np.sum(original_order_weights))
    print(f"Weights shape: {original_order_weights.shape}")
    print(f"Min weight: {original_order_weights.min():.4f}, Max weight: {original_order_weights.max():.4f}")
    print(f"Mean weight: {original_order_weights.mean():.4f}")

    true_angular_range = np.sum(weights) # this may not be the same as the max angular range if we have gaps
    # Normalize weights:
    num_unique_angles = len(angles)-total_duplicates
    original_order_weights = original_order_weights  / (np.sum(weights)/len(weights))
    # / (true_angular_range/num_unique_angles) # used to do this
    print("After normalization:")
    print("Total of weights: ", np.sum(original_order_weights))
    print(f"Weights shape: {original_order_weights.shape}")
    print(f"Min weight: {original_order_weights.min():.4f}, Max weight: {original_order_weights.max():.4f}")
    print(f"Mean weight: {original_order_weights.mean():.4f}")


    print("Num unique angles: ", num_unique_angles)
    print("Num projections: ", num_proj)
    print("True angular range:", true_angular_range, np.radians(true_angular_range))


    return original_order_weights

def get_angular_coverage(self):
    gaps_no_zeros

def normalise_weights_for_FBP(weights):

    normalised_weights = weights.copy()
    normalised_weights = normalised_weights / np.sum(normalised_weights) * len(normalised_weights)
    return normalised_weights







# def run_weighted_fbp(data, weights):
#     # Apply weights to the data
#     data_weighted = data.copy()
#     for angle_idx in range(data.geometry.angles.size):
#         try:
#             data_weighted.array[angle_idx, :, :] *= weights[angle_idx]
#         except:
#             data_weighted.array[angle_idx, :] *= weights[angle_idx]

#     # Reorder and reconstruct with weighted data
#     data_weighted.reorder('astra')
#     recon = FBP_ASTRA(data_weighted.geometry.get_ImageGeometry(), data_weighted.geometry)(data_weighted)
#     return recon


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
