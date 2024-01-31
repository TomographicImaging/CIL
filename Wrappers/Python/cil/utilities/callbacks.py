from abc import ABC, abstractmethod
from functools import partialmethod

from tqdm.auto import tqdm as tqdm_auto
from tqdm.std import tqdm as tqdm_std


class Callback(ABC):
    '''Base Callback to inherit from for use in :code:`Algorithm.run(callbacks: list[Callback])`.

    Parameters
    ----------
    verbose: int, choice of 0,1,2, default 1
        0=quiet, 1=info, 2=debug.
    '''
    def __init__(self, verbose=1):
        self.verbose = verbose

    @abstractmethod
    def __call__(self, algorithm):
        pass


class _OldCallback(Callback):
    '''Converts an old-style :code:`def callback` to a new-style :code:`class Callback`.

    Parameters
    ----------
    callback: :code:`callable(iteration, objective, x)`
    '''
    def __init__(self, callback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = callback

    def __call__(self, algorithm):
        if algorithm.update_objective_interval > 0 and algorithm.iteration % algorithm.update_objective_interval == 0:
            self.func(algorithm.iteration, algorithm.get_last_objective(return_all=self.verbose>=2), algorithm.x)


class ProgressCallback(Callback):
    ''':code:`tqdm`-based progress bar.

    Parameters
    ----------
    tqdm_class: default :code:`tqdm.auto.tqdm`
    **tqdm_kwargs:
        Passed to :code:`tqdm_class`.
    '''
    def __init__(self, verbose=1, tqdm_class=tqdm_auto, **tqdm_kwargs):
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
    ''':code:`tqdm`-based progress but text-only updates on separate lines.

    Parameters
    ----------
    bar_format: str
        Passed to :code:`tqdm`.
    '''
    def __init__(self, *args, bar_format="{n:>6d}/{total_fmt:<6} {rate_fmt:>9}{postfix}", **kwargs):
        super().__init__(*args, bar_format=bar_format, mininterval=0, maxinterval=0, position=0, **kwargs)
        self._instances.remove(self)  # don't interfere with external progress bars

    @staticmethod
    def status_printer(file):
        fp_flush = getattr(file, 'flush', lambda: None)

        def fp_write(s):
            file.write(f"{s}\n")
            fp_flush()

        return fp_write


class TextProgressCallback(ProgressCallback):
    ''':code:`ProgressCallback` but printed on separate lines to screen.

    Parameters
    ----------
    miniters: int, default :code:`Algorithm.update_objective_interval`
        Number of algorithm iterations between screen prints.
    '''
    __init__ = partialmethod(ProgressCallback.__init__, tqdm_class=_TqdmText)

    def __call__(self, algorithm):
        if not hasattr(self, 'pbar'):
            self.tqdm_kwargs['miniters'] = min((
                self.tqdm_kwargs.get('miniters', algorithm.update_objective_interval),
                algorithm.update_objective_interval))
        return super().__call__(algorithm)


class LogfileCallback(TextProgressCallback):
    ''':code:`TextProgressCallback` but to a file instead of screen.

    Parameters
    ----------
    log_file: FileDescriptorOrPath
        Passed to :code:`open()`.
    mode: str
        Passed to :code:`open()`.
    '''
    def __init__(self, log_file, mode='a', **kwargs):
        self.fd = open(log_file, mode=mode)
        super().__init__(file=self.fd, **kwargs)
