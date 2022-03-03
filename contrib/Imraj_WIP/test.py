from tensorboardX import SummaryWriter
import numpy as np
writer = SummaryWriter()


for n_iter in range(100):

    dummy_s1 = np.random.rand(1)
    dummy_s2 = np.random.rand(1)
    # data grouping by `slash`
    writer.add_scalar('data/scalar1', dummy_s1[0], n_iter)
    writer.add_scalar('data/scalar2', dummy_s2[0], n_iter)

    writer.add_scalars('data/scalar_group', {'xsinx': n_iter * np.sin(n_iter),
                                             'xcosx': n_iter * np.cos(n_iter),
                                             'arctanx': np.arctan(n_iter)}, n_iter)