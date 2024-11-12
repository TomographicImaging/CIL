from abc import ABC, abstractmethod
from datetime import datetime
from functools import partialmethod
from os.path import join
from pathlib import Path
from sqlite3 import Connection
from typing import Callable, List, Union

import numpy as np
from tqdm.auto import tqdm as tqdm_auto
from tqdm.std import tqdm as tqdm_std

from ..algorithms import Algorithm


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
        self._obj_len = 0  # number of objective updates

    def __call__(self, algorithm):
        if not hasattr(self, 'pbar'):
            tqdm_kwargs = self.tqdm_kwargs
            tqdm_kwargs.setdefault('total', algorithm.max_iteration)
            tqdm_kwargs.setdefault('disable', not self.verbose)
            tqdm_kwargs.setdefault('initial', max(0, algorithm.iteration))
            self.pbar = self.tqdm_class(**tqdm_kwargs)
        if (obj_len := len(algorithm.objective)) != self._obj_len:
            self.pbar.set_postfix(algorithm.objective_to_dict(self.verbose>=2), refresh=False)
            self._obj_len = obj_len
        self.pbar.update(algorithm.iteration - self.pbar.n)


class _TqdmText(tqdm_std):
    ''':code:`tqdm`-based progress but text-only updates on separate lines.

    Parameters
    ----------
    num_format: str
        Format spec for postfix numbers (i.e. objective values).
    bar_format: str
        Passed to :code:`tqdm`.
    '''
    def __init__(self, *args, num_format='+8.3e', bar_format="{n:>6d}/{total_fmt:<6} {rate_fmt:>9}{postfix}", **kwargs):
        self.num_format = num_format
        super().__init__(*args, bar_format=bar_format, mininterval=0, maxinterval=0, position=0, **kwargs)
        self._instances.remove(self)  # don't interfere with external progress bars

    @staticmethod
    def status_printer(file):
        fp_flush = getattr(file, 'flush', lambda: None)

        def fp_write(s):
            file.write(f"{s}\n")
            fp_flush()

        return fp_write

    def format_num(self, n):
        return f'{n:{self.num_format}}'

    def display(self, *args, **kwargs):
        """
        Clears :code:`postfix` if :code:`super().display()` succeeds
        (if display updates are more frequent than objective updates, users should not think the objective has stabilised).
        """
        if (updated := super().display(*args, **kwargs)):
            self.set_postfix_str('', refresh=False)
        return updated


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
        
class EarlyStoppingObjectiveValue(Callback):
    '''Callback that stops iterations if the change in the objective value is less than a provided threshold value.

    Parameters
    ----------
    threshold: float, default 1e-6 

    Note
    -----
    This callback only compares the last two calculated objective values. If `update_objective_interval` is greater than 1, the objective value is not calculated at each iteration (which is the default behaviour), only every `update_objective_interval` iterations.
    
        '''
    def __init__(self, threshold=1e-6):
        self.threshold=threshold
    
    
    def __call__(self, algorithm):
        if len(algorithm.loss)>=2:
            if np.abs(algorithm.loss[-1]-algorithm.loss[-2])<self.threshold:
                raise StopIteration
                
class CGLSEarlyStopping(Callback):
    '''Callback to work with CGLS. It causes the algorithm to terminate if  :math:`||A^T(Ax-b)||_2 < \epsilon||A^T(Ax_0-b)||_2` where `epsilon` is set to default as '1e-6', :math:`x` is the current iterate and :math:`x_0` is the initial value. 
    It will also terminate if the algorithm begins to diverge i.e. if :math:`||x||_2> \omega`, where `omega` is set to default as 1e6. 
    Parameters
    ----------
    epsilon: float, default 1e-6 
        Usually a small number: the algorithm to terminate if :math:`||A^T(Ax-b)||_2 < \epsilon||A^T(Ax_0-b)||_2`
    omega: float, default 1e6 
        Usually a large number: the algorithm will terminate if  :math:`||x||_2> \omega`
        
    Note
    -----
    This callback is implemented to replicate the automatic behaviour of CGLS in CIL versions <=24. It also replicates the behaviour of https://web.stanford.edu/group/SOL/software/cgls/. 
    '''
    def __init__(self, epsilon=1e-6, omega=1e6):
        self.epsilon=epsilon
        self.omega=omega
    
    
    def __call__(self, algorithm):
        
        if (algorithm.norms <= algorithm.norms0 * self.epsilon):
            print('The norm of the residual is less than {} times the norm of the initial residual and so the algorithm is terminated'.format(self.epsilon))
            raise StopIteration
        self.normx = algorithm.x.norm()
        if algorithm.normx >= self.omega:
            print('The norm of the solution is greater than {} and so the algorithm is terminated'.format(self.omega))
            raise StopIteration
            
        
class SqliteCallback(Callback):
    """
    Callback which evaluates user-defined functions art user-defined times during evaluation
    Parameters
    ----------
    db_address : Union[str, Path]
        Location of the sqlite database we're going to write to.
    output_folder : Union[str, Path]
        Location of folder to which artifacts will be written.
    report_progress : Callable[[Algorithm], bool]
        Rules which tells us when to report calculation progress using the functions in to_store. Note that
        this is in union with grab_soln's reports.
    grab_soln : Callable[[Algorithm], bool]
        Rule which tells us when to bother saving solution state.
    to_store : List[Callable[[Algorithm], Union[str, int, float]]]
        List of functions to get evaluated and stored.
    types : List[str]
        Return SQL types (TEXT, INT, etc.) of the to_store functions.
    names : List[str]
        Names of columns for entries to be recorded in database.
    calculation_index : int
        Index assigned to this calculation run.
    table_name : str
        Name of table we're going to put results in.
    verbose : int
        Verbosity for Callback.
    """
    def __init__(self, db_address: Union[str, Path], output_folder: Union[str, Path],
                 report_progress: Callable[[Algorithm], bool], grab_soln: Callable[[Algorithm], bool],
                 to_store: List[Callable[[Algorithm], Union[str, int, float]]], types: List[str],
                 names: List[str], calculation_index: int, table_name: str = "cil_calc_results",
                 verbose: int = 1):
        self._db_address = Path(db_address).resolve()
        self._output_folder = Path(output_folder).resolve()
        self._take_snapshot = grab_soln
        self._to_store = to_store
        self._types = types
        self._names = names
        self._calculation_label = calculation_index
        self._table_name = table_name
        self._report_progress = report_progress
        self._insert_heading_command = f"(time_stamp, calc_label, iteration, {', '.join(self._names)}, artifact_location)"
        super().__init__(verbose=verbose)
        variable_length_column_definition = ", ".join([f"{self._names[output_index]} " +
                                                       f"{self._types[output_index]}"
                                                       for output_index in range(len(self._to_store))])
        con = Connection(self._db_address)
        cur = con.cursor()
        cur.execute(f"CREATE TABLE IF NOT EXISTS {self._table_name}(id INTEGER PRIMARY KEY, time_stamp TEXT, " +
                    f"calc_label INT, iteration INT, {variable_length_column_definition}, artifact_location TEXT);")
        con.commit()
        con.close()

    def __call__(self, algorithm: Algorithm):
        artifact_location = "NULL"
        take_snapshot = self._take_snapshot(algorithm)
        report_progress = self._report_progress(algorithm)
        iteration = algorithm.iteration
        if take_snapshot:
            artifact_location = join(self._output_folder, f"{self._calculation_label}_{iteration}.npy")
            np.save(artifact_location, algorithm.solution.array)
            artifact_location = f"\'{artifact_location}\'"
        if take_snapshot or report_progress:
            storables = [str(f(algorithm)) for f in self._to_store]
            current_time = str(datetime.now())
            runstring = (f"INSERT INTO {self._table_name}{self._insert_heading_command} " +
                         f"VALUES(\'{current_time}\', {self._calculation_label}, {iteration}, " +
                         f"{', '.join(storables)}, {artifact_location});")
            con = Connection(self._db_address)
            cur = con.cursor()
            cur.execute(runstring)
            con.commit()
            con.close()