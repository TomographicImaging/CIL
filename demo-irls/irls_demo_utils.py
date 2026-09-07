"""
Data, scoring, tables and figures for ``irls_demo.ipynb``.

Deliberately nothing about the algorithms. How the regularising operator, the
inner solver, IRLS and FISTA are set up is what the demo is for, so all of that
stays in the notebook where it can be read. What lives here is everything
around it: loading and preprocessing the sphere, one objective both methods can
be scored against, a runner, and the table and figure formatting.
"""

from collections import namedtuple

import numpy as np
from time import time

from cil.plugins.astra import ProjectionOperator
from cil.processors import Slicer, TransmissionAbsorptionConverter
from cil.recon import FDK
from cil.utilities import dataexample
from cil.utilities.display import show2D


# --------------------------------------------------------------------------- #
# the shared objective, and scoring against it
# --------------------------------------------------------------------------- #

def l1(x):
    """Sum of absolute values, recursing into a BlockDataContainer."""
    if hasattr(x, 'containers'):
        return sum(l1(c) for c in x.containers)
    return float(np.abs(x.as_array()).sum())


def make_objective(A, data, L, alpha):
    r"""
    ``||Au-b||^2 + alpha^2 ||Lu||_1``, with :math:`L = I` for ``None``.

    Both methods have to be scored from the outside. IRLS reports the inner
    solver's residual against a *different reweighted* operator every outer
    iteration, so its own ``objective`` is not a fixed quantity and is not
    comparable with FISTA's.
    """
    def objective(u):
        residual = A.direct(u) - data
        return (float(residual.squared_norm())
                + alpha ** 2 * l1(u if L is None else L.direct(u)))
    return objective


def relative_rmse(u, ground_truth):
    """Relative RMSE against the ground truth, over the whole container."""
    return float((u - ground_truth).norm() / ground_truth.norm())


Result = namedtuple('Result', 'label solution objective rmse')


def run_variants(variants, objective, ground_truth, name, prefix=''):
    """
    Run each ``(label, build, iterations)`` to its full budget and score it.

    ``build`` is a callable rather than an algorithm so that nothing is
    constructed until it is about to be run -- at 3D sizes the set-up cost of
    an inner solver is not free. ``label`` may be a callable too, taking the
    built algorithm, so that a label can report what the algorithm resolved to
    -- such as its ``form`` -- rather than predicting it.
    """
    results = []
    for label, build, iterations in variants:
        algorithm = build()
        if callable(label):
            label = label(algorithm)
        t1 = time()
        algorithm.run(iterations, verbose=0)
        t2 = time()
        print(f'Time taken for {label} ({name}): {t2 - t1:.2f} seconds')
        solution = algorithm.get_output().copy()
        result = Result(label, solution, objective(solution),
                        relative_rmse(solution, ground_truth))
        print('  {:<44} objective {:.5e}  rmse {:.4e}'
              .format('{}{} ({})'.format(prefix, label, name),
                      result.objective, result.rmse))
        results.append(result)
    return results


# --------------------------------------------------------------------------- #
# figures and tables
# --------------------------------------------------------------------------- #

def show_row(panels, title, display=None):
    """
    One row of images, ``panels`` being ``(label, container)`` pairs.

    ``display`` maps a container to the 2D array to draw; pass
    ``centre_slice`` from the data loader for a volume. ``show2D`` calls
    ``plt.show()`` itself, so the row is rendered as it is built and the title
    goes above it rather than as a suptitle.
    """
    if not panels:
        return None
    draw = display or (lambda container: container)
    print(title)
    return show2D([draw(container) for _, container in panels],
                  title=[label for label, _ in panels],
                  origin='upper', num_cols=len(panels),
                  size=(3.2 * len(panels), 3.6))


def table(rows, columns):
    """A plain text table, printed with the columns aligned."""
    widths = [max(len(str(row[i])) for row in [columns] + rows)
              for i in range(len(columns))]
    line = '  '.join('{:<%d}' % width for width in widths)
    print('\n' + line.format(*columns))
    print('  '.join('-' * width for width in widths))
    for entry in rows:
        print(line.format(*(str(cell) for cell in entry)))


def row(result, *labels):
    """A table row: some labels, then the numbers from a result."""
    return list(labels) + ['{:.5e}'.format(result.objective),
                           '{:.4e}'.format(result.rmse)]

# --------------------------------------------------------------------------- #
# the data
# --------------------------------------------------------------------------- #

def load_and_process_sphere(angle_step: int = 5, single_slice: bool = False):
    """
    Loads and preprocesses the sphere dataset.

    Parameters
    ----------
    angle_step : int, optional
        The step size for angular subsampling. Default is 5.
    single_slice : bool, optional
        If True, extracts and processes only the central 2D slice. 
        If False, processes the full 3D volume. Default is False.

    Returns
    -------
    absorption : DataContainer
        The processed CIL DataContainer (absorption data).
    A : ProjectionOperator
        The Astra ProjectionOperator corresponding to the data geometry.
    ig : ImageGeometry
        The ImageGeometry of the dataset.
    ground_truth : ImageData
        The ground truth volume/slice.
    recon : ImageData
        The FDK reconstructed volume/slice.
    """
    
    # Load data
    ground_truth = dataexample.SIMULATED_SPHERE_VOLUME.get()
    data = dataexample.SIMULATED_CONE_BEAM_DATA.get()

    # Conditionally extract a single 2D slice
    if single_slice:
        data = data.get_slice(vertical='centre')
        ground_truth = ground_truth.get_slice(vertical='centre')

    # Convert to absorption and subsample angles
    absorption = TransmissionAbsorptionConverter()(data)
    absorption = Slicer(roi={'angle': (0, -1, angle_step)})(absorption)
    
    # Reconstruct the initial guess using TIGRE's FDK
    absorption.reorder('tigre')
    ig_tigre = ground_truth.geometry
    recon = FDK(absorption, image_geometry=ig_tigre).run()

    # Reorder data to match Astra toolbox
    absorption.reorder('astra')
    ground_truth.reorder('astra')
    recon.reorder('astra')
    
    # Grab the updated geometry after reordering
    ig_astra = ground_truth.geometry

    # Visualization
    if single_slice:
        show2D([ground_truth, recon], 
               title=['Ground Truth', 'FDK Reconstruction'], 
               origin='upper', num_cols=2)
    else:
        # If 3D, grab the central slice just for the display plot
        show2D([ground_truth.get_slice(vertical='centre'), recon.get_slice(vertical='centre')], 
               title=['Ground Truth (Central Slice)', 'FDK Reconstruction (Central Slice)'], 
               origin='upper', num_cols=2)

    # Setup Astra Projection Operator
    A = ProjectionOperator(image_geometry=ig_astra, acquisition_geometry=absorption.geometry)
    
    return absorption, A, ig_astra, ground_truth, recon

def centre_slice(x):
    """A 2D array to draw: the container itself in 2D, its centre slice in 3D."""
    array = x.as_array()
    return array if array.ndim == 2 else array[array.shape[0] // 2]