from functools import wraps
from time import perf_counter_ns

import cProfile, pstats

def time_this(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter_ns()
        result = func(*args, **kwargs)
        end = perf_counter_ns()
        print(func.__name__, f"{(end - start) * 1e-09:.6f}s")
        return result
    return wrapper

class Profiler:
    def __init__(self, func, sort_stats_by):
        self.func = func
        self.profile_runs = []
        self.sort_stats_by = sort_stats_by

    def __call__(self, *args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = self.func(*args, **kwargs)
        profiler.disable()
        self.profile_runs.append(profiler)
        stats = pstats.Stats(*self.profile_runs).sort_stats(self.sort_stats_by)
        return result, stats

def cumulative_profiler(runs, sort_stats_by='time', percent_of_lines=0.05):
    def decorator(function):
        def wrapper(*args, **kwargs):
            print(f'\nTotal runs to perform: {runs}')
            profiler = Profiler(function, sort_stats_by)
            for _ in range(runs):
                result, stats = profiler(*args, **kwargs)
            stats.print_stats(percent_of_lines)
            return result
        return wrapper

    if callable(runs):
        default_runs = runs
        runs = 5 # default value
        return decorator(default_runs)
    return decorator
