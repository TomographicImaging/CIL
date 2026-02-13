import numpy as np

from cil.plugins.astra import FBP as FBP_ASTRA
import numba
from cil.framework.labels import AngleUnit

def get_weights_for_FBP(data):

    angles = data.geometry.angles.copy()

    angle_range = np.max(angles) - np.min(angles)

    if data.geometry.config.angles.angle_unit == AngleUnit.DEGREE:
        half_angle = 180.0
    else:
        half_angle = np.pi

    if angle_range > half_angle*1.05: # TODO: decide how to establish this
        angle_range = half_angle*2.0
    else:
        angle_range = half_angle


    # Create array of angle indices in order of angle values
    sorted_indices = np.argsort(angles)
    ordered_angles = angles[sorted_indices]

    num_angles = len(ordered_angles)
    weights = np.zeros(num_angles)
    

    # create tolerance for two angles being equal:
    gaps=[]
    for i in range(num_angles):
        prev_idx = (i - 1) % num_angles
        gaps.append((ordered_angles[i]-ordered_angles[prev_idx]) % num_angles)
    mean_gap = np.mean(np.asarray(gaps))

    angular_tolerance = 0.1*mean_gap

    # check if all gaps are equal, if so return weights of 1:
    if np.allclose(gaps, mean_gap, atol=angular_tolerance):
        return np.ones_like(angles)   

    num_proj_at_same_angle = 1
    total_duplicates=0

    for i in range(num_angles):

        # skips if current angle same as prev angle
        # as has already been assigned a weight:
        if num_proj_at_same_angle >1:
            num_proj_at_same_angle -= 1
            continue

        prev_idx = (i - 1) % num_angles
        next_idx = (i + 1) % num_angles

        prev_angle = ordered_angles[prev_idx]
        next_angle = ordered_angles[next_idx]
        current_angle = ordered_angles[i]

        # if weight is zero, need to change approach:
        # if we have two angles that are the same:
        # then they need to be the weight of the following angle - the previous angle, divided by 2.0
        # but need to see how many angles are the same, and then divide by that number as well

        
        if np.isclose(next_angle, current_angle, atol=angular_tolerance):
            while np.isclose(next_angle, current_angle, atol=angular_tolerance):
                next_idx = (next_idx + 1) % num_angles
                next_angle = ordered_angles[next_idx]
                num_proj_at_same_angle += 1
            for j in range(num_proj_at_same_angle):
                weights[i+j] = (next_angle-prev_angle) % angle_range / (2.0*num_proj_at_same_angle)
            total_duplicates+=num_proj_at_same_angle-1

        else:
            weights[i] = (next_angle-prev_angle) % angle_range / 2.0

    # Reorder weights back to original projection order
    original_order_weights = np.zeros(num_angles)
    for i, idx in enumerate(sorted_indices):
        original_order_weights[idx] = weights[i]

    print("Before normalization:")
    print("Total of weights: ", np.sum(original_order_weights))
    print(f"Weights shape: {original_order_weights.shape}")
    print(f"Min weight: {original_order_weights.min():.4f}, Max weight: {original_order_weights.max():.4f}")
    print(f"Mean weight: {original_order_weights.mean():.4f}")


    # Normalize weights:
    original_order_weights = original_order_weights / (angle_range/(len(angles)-total_duplicates))
    print("After normalization:")
    print("Total of weights: ", np.sum(original_order_weights))
    print(f"Weights shape: {original_order_weights.shape}")
    print(f"Min weight: {original_order_weights.min():.4f}, Max weight: {original_order_weights.max():.4f}")
    print(f"Mean weight: {original_order_weights.mean():.4f}")

    return original_order_weights

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

    proj_size = v_size*h_size
    num_proj = int(data.array.size / proj_size)

    data_weighted = data.copy() # Doesn't alter the original AcquisitionData
    data_weighted_array = data_weighted.array

    data_weighted_reshaped = data_weighted_array.reshape(num_proj, proj_size)
    

    if accelerated:
        numba_loop(weights, num_proj, proj_size, data_weighted_array)
    
    else:
        weights = np.asarray(weights)[:, np.newaxis] # shape: (num_proj, 1) 
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
