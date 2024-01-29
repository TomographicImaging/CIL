
from abc import ABC, abstractmethod

from tqdm.auto import tqdm as tqdm_auto
from tqdm.std import tqdm as tqdm_std


class Callback(ABC):
    '''Base Callback to inherit from for use in :code:`Algorithm.run(callbacks: list[Callback])`.'''
    def __init__(self, verbose=1):
        ''':param verbose: int, 0=quiet, 1=info, 2=debug'''
        self.verbose = verbose

    @abstractmethod
    def __call__(self, algorithm):
        pass


class OldCallback(Callback):
    '''Converts an old-style :code:`def callback(iteration, objective, x)`
    to a new-style :code:`class Callback`.
    '''
    def __init__(self, callback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = callback

    def __call__(self, algorithm):
        if algorithm.update_objective_interval > 0 and algorithm.iteration % algorithm.update_objective_interval == 0:
            self.func(algorithm.iteration, algorithm.get_last_objective(return_all=self.verbose), algorithm.x)


class ProgressCallback(Callback):
    ''':code:`tqdm`-based progress bar.'''
    def __init__(self, verbose=1, tqdm_class=tqdm_auto, **tqdm_kwargs):
        '''
        :param tqdm_kwargs: passed to :code:`tqdm_class`.
        '''
        super().__init__(verbose=verbose)
        self.tqdm_class = tqdm_class
        self.tqdm_kwargs = tqdm_kwargs

    def __call__(self, algorithm):
        if not hasattr(self, 'pbar'):
            tqdm_kwargs = self.tqdm_kwargs
            tqdm_kwargs.setdefault('total', algorithm.max_iteration)
            tqdm_kwargs.setdefault('disable', not self.verbose)
            self.pbar = self.tqdm_class(**tqdm_kwargs)
        self.pbar.set_postfix(algorithm.objective_to_dict(self.verbose>=2), refresh=False)
        self.pbar.update(algorithm.iteration - self.pbar.n)


class _TqdmText(tqdm_std):
    ''':code:`tqdm`-based progress but text-only updates on separate lines.'''
    def __init__(self, *args, mininterval=1, position=0, **kwargs):
        """:param miniterval: approximate number of seconds between updates."""
        super().__init__(*args, mininterval=mininterval, position=position, **kwargs)
        self._instances.remove(self)  # don't interfere with external progress bars

    @staticmethod
    def status_printer(file):
        fp_flush = getattr(file, 'flush', lambda: None)

        def fp_write(s):
            file.write(f"{s}\n")
            fp_flush()

        return fp_write


class TextProgressCallback(ProgressCallback):
    ''':code:`ProgressCallback` but printed on separate lines to screen.'''
    def __init__(self, tqdm_class=_TqdmText, **kwargs):
        """:param miniterval: approximate number of seconds between updates."""
        super().__init__(tqdm_class=tqdm_class, **kwargs)


class LogfileCallback(TextProgressCallback):
    ''':code:`TextProgressCallback` but to a file instead of screen.'''
    def __init__(self, log_file, mode='a', **kwargs):
        """
        :param log_file: passed to :code:`open()`.
        :param mode: passed to :code:`open()`.
        """
        self.fd = open(log_file, mode=mode)
        super().__init__(file=self.fd, **kwargs)
