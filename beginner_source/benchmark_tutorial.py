"""
PyTorch Benchmark
=======================
This recipe explains how to use PyTorch's benchmark module to measure
the performance of your code and compare it against other code.

Introduction
------------
Benchmarking is an important step in writing new code and an essential
step when optimizing/refactoring existing code to prevent performance
regressions.

There are many benchmarking tools out there including the python builtin
module timeit. However, benchmarking PyTorch code has many caveats that
are easy to forget when using these modules, such as managing number of
threads and synchronizing CUDA devices. And any good benchmark will run a
set of different input parameters, but creating different input tensors
can be quite tedious.

In this recipe we will benchmark a custom implementation of a PyTorch
function against the builtin version. In the process we will learn to
use PyTorch benchmark to handle the many caveats of benchmarking, to
compare performance of different versions and to generate a set of input
parameters to benchmark against.

Setup
-----
Before we begin, we need to install ``torch`` if it isn’t already available.
https://pytorch.org/get-started/locally/

::

   pip install torch

      or

   conda install pytorch -c pytorch

"""

######################################################################
# Steps
# -----
# 
# 1. Defining some helper functions
# 2. Writing the custom C++ extension
# 3. Simple benchmark using timeit.Timer
# 4. Better benchmark using torch.benchmark.Timer
# 4.1. Timer.blocked_autorage
# 4.2. Why runtime awareness matters
# 5. Comparing different benchmarks
# 6. Generating bechmark inputs with Fuzzed Parameters
# 6.1. Canned fuzzers: back to our x+1 kernel
# 7. Taking more precise measurements with instruction counting
# 
# 1. Defining some helper functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# We'll start by defining some helper functions which we'll use later.
#

import os
import collections
import textwrap
import re
from typing import List, Union

from IPython.display import Markdown, display
import matplotlib.pyplot as plt

import torch
import numpy as np


# We want to show certain threading effects, but 1 vs. several dozen
# is often too stark a contrast.
torch.set_num_threads(4)


def print_as_cpp(source: str):
    display(Markdown(f"```c++\n{source}\n```"))


def load_extension(name: str, code: str, fn_name: str):
   """Compile our implementation into an inline module.

   Normally we would modify ATen instead, however this allows us
   to show an example without having to build PyTorch from source.
   """
   from torch.utils.cpp_extension import load_inline
   return load_inline(
      name,
      code,
      extra_cflags=["-O2", "-g", "-mavx2", "-mfma"],
      functions=[fn_name])


def module_to_setup_str(m):
   """Handle importing `m` during Timer setup.

   This step is only necessary because we are using custom extensions for
   demonstration, rather than modifying and rebuilding PyTorch core.
   """
   module_dir, module_name = os.path.split(m.__file__)
   return textwrap.dedent(f"""
      import sys
      if not {repr(module_dir)} in sys.path:
         sys.path.append({repr(module_dir)})
      import {module_name[:-3]} as my_module
      """)

######################################################################
# 2. Writing the custom C++ extension
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Next, we are going to write a custom C++ extension to implement
# the ``x + 1`` operation for some tensor x. We are going to use this
# as a case study for how to take a systematic approach towards
# optimizing it. For simplicity, we will only consider float Tensors
# on CPU.
#

shift_impl_v0_src = """
// First attempt at a specialized implementation of `x + 1`
at::Tensor shift(const at::Tensor& x) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "shift requires a float input");

    auto y = x.clone();
    auto y_ptr = y.data_ptr<float>();
    auto n = y.numel();
    for (int64_t i = 0; i < n; ++i) {
       *(y_ptr + i) += 1;
    }

    return y;
}
"""

# Output hidden
print_as_cpp(shift_impl_v0_src)
shift_impl_v0 = load_extension('shift_impl_v0', shift_impl_v0_src, 'shift')

######################################################################
# 3. Simple benchmark using timeit.Timer
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Let's have a first attempt at benchmarking the above extension
# using the builint timeit module in python.
#

import timeit

repeats = 5
sizes = (1, 1024, 16384)


def measure_native(n):
    num_runs, total_time = timeit.Timer(
        "x + 1", 
        setup=f"import torch; x = torch.ones(({n},))",
    ).autorange()
    return total_time / num_runs


def measure_cpp(n):
    num_runs, total_time = timeit.Timer(
        "shift(x)", 
        setup=f"import torch; x = torch.ones(({n},))",
        globals={"shift": shift_impl_v0.shift},
    ).autorange()
    return total_time / num_runs


for title, measure_fn in (("Native", measure_native), ("\n\nC++ Extension", measure_cpp)):
    print(f"{title}\n" + "".join([f"n = {i}".rjust(13) for i in sizes]) + "\n" + "-" * 13 * len(sizes))
    for _ in range(repeats):
        result_line = ""
        for n in sizes:
            result_line += f"{measure_fn(n) * 1e6:10.1f} us"
        print(result_line)

# Native
#         n = 1     n = 1024    n = 16384
# ---------------------------------------
#        8.3 us       8.9 us      14.5 us
#        8.2 us       8.8 us      14.4 us
#        8.4 us       8.7 us      13.8 us
#        8.3 us       8.5 us      14.0 us
#        8.1 us       8.6 us      14.1 us
#
#
# C++ Extension
#         n = 1     n = 1024    n = 16384
# ---------------------------------------
#        4.1 us       5.1 us      17.8 us
#        4.2 us       5.1 us      21.7 us
#        4.1 us       5.0 us      17.9 us
#        4.1 us       5.0 us      18.4 us
#        4.1 us       5.1 us      20.7 us

######################################################################
# 4. Better benchmark using torch.benchmark.Timer
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Now we will see how we can use PyTorch's benchmark module to improve
# our benchmark above.
#

from torch.utils.benchmark import Timer


timer = Timer(
    stmt="x + 1",
    setup="x = torch.ones((1,))",
)

# The torch utils Timer returns a Measurement object, which contains
# metadata about the run as well as replicates, if applicable.
print(timer.timeit(100), "\n")


m = Timer(
    stmt="x + 1",
    
    # Like timeit.Timer, initialization can be done using `setup=...`
    # or `globals=...` (or both).
    globals={"x": torch.ones((1,))},
    
    # torch.utils.benchmark.Timer takes several additional annotation argument:
    #   label, sub_label, description, and env
    # These change the __repr__ measurements, and are used when grouping and
    # displaying measurements. (Discussed later.)
    label="Add one",
    sub_label="Generic implementation.",
)

print(timer.timeit(100))

# <torch.utils.benchmark.utils.common.Measurement object at 0x7f46e18717f0>
# x + 1
#   11.19 us
#   1 measurement, 100 runs , 1 thread 
#
# <torch.utils.benchmark.utils.common.Measurement object at 0x7f46e579e588>
# x + 1
#   8.30 us
#   1 measurement, 100 runs , 1 thread

######################################################################
# 4.1. Timer.blocked_autorage
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# A mixture of timeit.Timer.repeat and timeit.Timer.autorange.
#
# While timeit.Timer.autorange takes a single continuous measurement
# of at least 0.2 seconds, torch.utils.benchmark.blocked_autorange
# takes many measurements whose times total at least 0.2 seconds
# (which can be changed by the min_run_time parameter) subject to the
# constraint that timing overhead is a small fraction of the overall
# measurement. This is acomplished by first running with an increasing
# number of runs per loop until the run time is much larger than
# measurement overhead (which also serves as a warm up), and then
# taking measurements until the target time is reached. This has the
# useful properties that it wastes less data, and allows us to take
# statistics in order to assess the reliability of measurements.
#

m = Timer(
    stmt="x + 1",
    setup="x = torch.ones((1,))",
).blocked_autorange()

# Print results summarized by __repr__
print(m, "\n")

# Print statistics
print(f"Mean:   {m.mean * 1e6:6.1f} us")
print(f"Median: {m.median * 1e6:6.1f} us")
print(f"IQR:    {m.iqr * 1e6:6.1f} us")
print(f"Times:  {str(m.times[:2])[:-1]}, ..., {str(m.times[-2:])[1:]}")

# <torch.utils.benchmark.utils.common.Measurement object at 0x7f46e182b9b0>
# x + 1
#   Median: 8.34 us
#   IQR:    0.28 us (8.23 to 8.51)
#   24 measurements, 1000 runs per measurement, 1 thread 
#
# Mean:      8.6 us
# Median:    8.3 us
# IQR:       0.3 us
# Times:  [8.493732661008834e-06, 8.615698665380478e-06, ..., 8.232343941926956e-06, 9.091291576623917e-06]

######################################################################
# 4.2. Why runtime awareness matters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# It's very easy to accidentally make an apples-to-oranges comparizon,
# such as comparing measurements with different numbers of threads,
# or forgetting to CUDA synchronize. The example below shows how to
# be explicit about multi-threaded analysis with PyTorch Benchmark.
#

x = torch.ones((1024, 1024))

num_runs, total_time = timeit.Timer("x + 1", globals={"x": x}).autorange()
m0 = Timer("x + 1", globals={"x": x}).blocked_autorange()
m1 = Timer("x + 1", globals={"x": x}, num_threads=torch.get_num_threads()).blocked_autorange()

print(f"timeit.Timer:                   {total_time / num_runs * 1e6:6.0f} us")
print(f"torch Timer:                    {m0.mean * 1e6:6.0f} us")
print(f"torch Timer(num_threads=...):   {m1.mean * 1e6:6.0f} us")

# timeit.Timer:                      111 us
# torch Timer:                       370 us
# torch Timer(num_threads=...):      106 us

######################################################################
# 5. Comparing different benchmarks
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# In the example below we show how to use the Compare feature of
# PyTorch benchmark to easily compare measurements from different runs.
#

from torch.utils.benchmark import Compare

results = []
for n in [1, 16, 256, 1024, 4096, 16384, 32768]:
    for num_threads in [1, 2, 4]:
        setup=f"x = torch.ones(({n},))"
        results.append(Timer(
            "x + 1",
            setup=setup,
            num_threads=num_threads,
            label="Shift operator",
            sub_label="Generic implementation.",
            description=str(n),
        ).blocked_autorange())

    results.append(Timer(
        "my_module.shift(x)",
        setup=(
            module_to_setup_str(shift_impl_v0) +
            setup
        ),
        # Custom C++ operator does not support parallelism.
        num_threads=1,
        label="Shift operator",
        sub_label="Custom C++ operator",
        description=str(n),
    ).blocked_autorange())

compare = Compare(results)
compare.print()

# [------------------------------------- Shift operator ------------------------------------]
#                                |   1   |   16  |  256  |  1024  |  4096  |  16384  |  32768
# 1 threads: --------------------------------------------------------------------------------
#       Generic implementation.  |  8.2  |  8.6  |  8.7  |  9.3   |  10.3  |   14.7  |   22.3
#       Custom C++ operator      |  4.1  |  4.2  |  4.5  |  5.3   |   8.2  |   18.6  |   35.3
# 2 threads: --------------------------------------------------------------------------------
#       Generic implementation.  |  8.0  |  8.4  |  8.7  |  9.3   |  10.4  |   14.6  |   23.4
# 4 threads: --------------------------------------------------------------------------------
#       Generic implementation.  |  7.9  |  8.7  |  8.9  |  9.3   |  10.5  |   14.6  |   23.8
#
# Times are in microseconds (us).

######################################################################
# With extra formatting (colors might not be shown here).
#

compare.trim_significant_figures()
compare.colorize()
compare.print()

# [------------------------------------- Shift operator ------------------------------------]
#                                |   1   |   16  |  256  |  1024  |  4096  |  16384  |  32768
# 1 threads: --------------------------------------------------------------------------------
#       Generic implementation.  |  8.2  |  8.6  |  8.7  |  9.3   |   10   |    15   |    22 
#       Custom C++ operator      |  4.1  |  4.2  |  4.5  |  5.3   |    8   |    19   |    35 
# 2 threads: --------------------------------------------------------------------------------
#       Generic implementation.  |  8.0  |  8.4  |  8.7  |  9.3   |   10   |    15   |    23 
# 4 threads: --------------------------------------------------------------------------------
#       Generic implementation.  |  7.9  |  8.7  |  8.9  |  9.3   |   11   |    15   |    24 
#
# Times are in microseconds (us).

######################################################################
# 6. Generating bechmark inputs with Fuzzed Parameters
#
# We'll take a brief detour and use fuzzed inputs to discuss transpose
# and contiguous before returning to shift.
#

from torch.utils.benchmark import Fuzzer, FuzzedParameter, FuzzedTensor, ParameterAlias

example_fuzzer = Fuzzer(
    parameters = [
        FuzzedParameter("k0", minval=1, maxval=1024 ** 2, distribution="loguniform"),
        FuzzedParameter("k1", distribution={1: 0.5, ParameterAlias("k0"): 0.5}, strict=True),
    ],
    tensors = [
        FuzzedTensor("x", size=("k0", "k1"), min_elements=128, max_elements=128 * 1024, probability_contiguous=0.6)
    ],
    seed=0,
)

results = []
for tensors, tensor_params, params in example_fuzzer.take(10):
    sub_label=f"{params['k0']:<6} x {params['k1']:<4} {'' if tensor_params['x']['is_contiguous'] else '(discontiguous)'}"
    for stmt in ("x.contiguous()", "x.t().contiguous()"):
        timer = Timer(
            stmt,
            globals=tensors, 
            label="2D transpose",
            description=stmt,
            sub_label=sub_label,
        )
        results.append(timer.blocked_autorange())

compare = Compare(results)
compare.trim_significant_figures()
compare.print()

# [------------------------------- 2D transpose ------------------------------]
#                                      |  x.contiguous()  |  x.t().contiguous()
# 1 threads: ------------------------------------------------------------------
#       355    x 355  (discontiguous)  |      200000      |          2000      
#       751    x 1                     |         200      |          2000      
#       313    x 313                   |         200      |        170000      
#       45851  x 1                     |         200      |          2000      
#       146    x 146                   |         200      |         43000      
#       15854  x 1                     |         200      |          2000      
#       143    x 143  (discontiguous)  |       40000      |          2000      
#       2709   x 1                     |         200      |          2000      
#       312    x 312                   |         200      |        170000      
#       5674   x 1                     |         200      |          2000      
#
# Times are in nanoseconds (ns).

######################################################################
# The fuzzed benchmarks reveal several noteworthy features:
#
# If a Tensor is already contiguous we do not need to construct a new
# Tensor and the operation is extremely cheap. O(100 ns)
#
# For N x 1 Tensors transpose requires that we create a new Tensor,
# but we can reuse the same buffer.
#
# For N x N tensors, either contiguous or transposed contiguous will
# be expensive depending on the underlying data layout.

######################################################################
# 6.1. Canned fuzzers: back to our x+1 kernel
#
# When benchmarking an op, there are a lot of things to consider:
# Dimensionality, contiguity (both layout and strides from slicing),
# broacasting, sizes, etc. While it's certainly possible to write your
# own fuzzer, it's nice if one doesn't have to. To that end, canned
# fuzzers are provided for unary and binary ops, and more will be added soon.
#

from torch.utils.benchmark.op_fuzzers import unary

def skip_if(x):
    # We know that around this point our custom kernel is slower than vanilla add, so
    # there's not much useful signal from testing larger Tensors.
    return x.numel() > 64 * 1024

def to_description(x, x_params):
    description = f"{str(list(x.shape)):<20}"
    order, steps = x_params["order"], x_params["steps"]
    
    order_str = ""
    if any(i != 1 for i in np.diff(order)):
        order_str = f"{tuple(order)}"
    description += order_str.ljust(30)
    
    if any(i != 1 for i in steps):
        description += f"[" + ", ".join([f"::{i}" if i > 1 else ":" for i in steps]) + "]"
    
    return description
    
results, descriptions, i = [], [], 0
for tensors, tensor_params, params in unary.UnaryOpFuzzer(seed=0).take(70):
    x = tensors["x"]
    if skip_if(x):
        continue
        
    descriptions.append(to_description(x, tensor_params["x"]))    
    timer = Timer(
        "x + 1",
        globals=tensors,
        label="Shift operator",
        sub_label="Generic: 'x + 1'",
        description=f"[{i}]",
    )
    results.append(timer.blocked_autorange(min_run_time=1))
    
    timer = Timer(
        "my_module.shift(x)",
        globals=tensors,
        setup=module_to_setup_str(shift_impl_v0),
        label="Shift operator",
        sub_label="Custom C++ operator",
        description=f"[{i}]",
    )
    results.append(timer.blocked_autorange(min_run_time=1))
    i += 1
    
compare = Compare(results)
compare.trim_significant_figures()
compare.colorize()
compare.print()

print(f"{'':<7}{'shape':<20}{'Layout permutation':<30}{'Steps from slicing'}")
for i, d in enumerate(descriptions):
    print(f"[{i}]".ljust(7) + d)

# [------------------------------------------------ Shift operator ------------------------------------------------]
#                            |  [0]  |  [1]  |  [2]  |  [3]  |  [4]  |  [5]  |  [6]  |  [7]  |  [8]  |  [9]  |  [10]
# 1 threads: -------------------------------------------------------------------------------------------------------
#       Generic: 'x + 1'     |   15  |   16  |   31  |   14  |   30  |   29  |   16  |   10  |   14  |   11  |   17 
#       Custom C++ operator  |   18  |   43  |   43  |   17  |   31  |   47  |   20  |   10  |   15  |   20  |   22 
#
# Times are in microseconds (us).
#
#        shape               Layout permutation            Steps from slicing
# [0]    [16384]                                           
# [1]    [38, 457]           (1, 0)                        
# [2]    [22, 35, 48]                                      [:, ::2, :]
# [3]    [14793]                                           
# [4]    [13544]                                           [::8]
# [5]    [325, 128]                                        [::2, :]
# [6]    [118, 166]                                        
# [7]    [6309]                                            
# [8]    [26, 512]                                         
# [9]    [27, 225]           (1, 0)                        
# [10]   [380, 58]                                         

######################################################################
# There's an interesting pattern in our benchmark results above. The
# dicontiguous input tensors [1], [2], [5] and [9] take much longer
# to run on than expected. The reason is that we are cloning the
# input tensor to contiguous memory. When the input tensor is 
# discontiguous this means we have to change the relative position
# of the elements in memory when copying which is slow. Below we
# will look at a new version that tries to tackle this issue.
#

shift_impl_v1_src = """
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

// Second attempt at a specialized implementation of x + 1
at::Tensor shift(const at::Tensor& x) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "shift requires a float input");

    auto result = at::empty_like(x);
    if (x.is_contiguous()) {
        // Fast path for contiguous inputs.
        auto x_ptr = x.data_ptr<float>();
        auto result_ptr = result.data_ptr<float>();
        auto n = x.numel();

        for (int64_t i = 0; i < n; ++i, ++x_ptr, ++result_ptr) {
           *result_ptr = *x_ptr + 1;
        }
    } else {
        // Fall back to more general machinery if x is discontiguous.
        auto iter = at::TensorIterator::unary_op(result, x);
        at::native::cpu_kernel(iter, [](float xi) -> float { return xi + 1; });
    }
    
    return result;
}
"""

# Output hidden
print_as_cpp(shift_impl_v1_src)
shift_impl_v1 = load_extension("shift_impl_v1", shift_impl_v1_src, "shift")

######################################################################
# Op fuzzers are deterministic.
#

i = 0
for tensors, tensor_params, params in unary.UnaryOpFuzzer(seed=0).take(70):
    x = tensors["x"]
    if skip_if(x):
        continue
    
    assert to_description(x, tensor_params["x"]) == descriptions[i]
    timer = Timer(
        "my_module.shift(x)",
        globals=tensors,
        setup=module_to_setup_str(shift_impl_v1),
        label="Shift operator",
        sub_label="Custom C++ (revised)",
        description=f"[{i}]",
    )
    results.append(timer.blocked_autorange(min_run_time=1))
    i += 1
    
compare = Compare(results)
compare.trim_significant_figures()
compare.colorize()
compare.print()

print(f"{'':<7}{'shape':<20}{'Layout permutation':<30}{'Steps from slicing'}")
for i, d in enumerate(descriptions):
    print(f"[{i}]".ljust(7) + d)

# [------------------------------------------------- Shift operator ------------------------------------------------]
#                             |  [0]  |  [1]  |  [2]  |  [3]  |  [4]  |  [5]  |  [6]  |  [7]  |  [8]  |  [9]  |  [10]
# 1 threads: --------------------------------------------------------------------------------------------------------
#       Generic: 'x + 1'      |   15  |   16  |   31  |   14  |   30  |   29  |   16  |   10  |   14  |   11  |   17 
#       Custom C++ operator   |   18  |   43  |   43  |   17  |   31  |   47  |   20  |   10  |   15  |   20  |   22 
#       Custom C++ (revised)  |   12  |   14  |   34  |   12  |   22  |   33  |   13  |    6  |   10  |    7  |   15 
#
# Times are in microseconds (us).
#
#        shape               Layout permutation            Steps from slicing
# [0]    [16384]                                           
# [1]    [38, 457]           (1, 0)                        
# [2]    [22, 35, 48]                                      [:, ::2, :]
# [3]    [14793]                                           
# [4]    [13544]                                           [::8]
# [5]    [325, 128]                                        [::2, :]
# [6]    [118, 166]                                        
# [7]    [6309]                                            
# [8]    [26, 512]                                         
# [9]    [27, 225]           (1, 0)                        
# [10]   [380, 58]                                         

######################################################################
# 7. Taking more precise measurements with instruction counting
#
# For the last part, we are going to take a look at how to get more
# accurate measurements using Callgrind for counting number of instructions
# performed instead of elapsed time. We will use an implementation that
# optimizes for a fast path case to compare the number of instructions
# executed in each case.
#

shift_impl_v2_src = f"""{shift_impl_v1_src.strip()}
at::Tensor shift_with_bailout(const at::Tensor& x) {{
    return x.numel() ? shift(x) : at::empty({0}, x.options());
}}
"""

print_as_cpp(shift_impl_v2_src)
shift_impl_v2 = load_extension("shift_impl_v2", shift_impl_v2_src, ["shift", "shift_with_bailout"])

def collect_counts_and_times(fn: str, n: int):
    timer = Timer(
        f"my_module.{fn}(x)",
        setup=(
            module_to_setup_str(shift_impl_v2) +
            f"x = torch.ones(({n},))"
        ))
    return (
        # Run for a long time to get robust statistics.
        timer.blocked_autorange(min_run_time=20),
        timer.collect_callgrind()
    )


def trim_count_repr(c):
    setup_str = textwrap.indent(module_to_setup_str(shift_impl_v2), " ")
    abridged_str = "\n import ... as my_module\n"
    return repr(c).replace(setup_str, abridged_str)


def delta(counts_0, counts_1):
    for c, fn in counts_1.as_standardized().delta(counts_0.as_standardized()):
        if "lookdict_unicode_nodummy" in fn:
            continue  # This is a noisy method in the Python interpreter.
            
        # Trim down some task specific prefixes to make them easier to read
        fn = re.sub(r"^.+torch_extensions/", "", fn)
        fn = re.sub(f"^.+site-packages/", "", fn)
        yield c, fn

def render_fast_path_effect(n: int):
    times, counts = collect_counts_and_times("shift", n)
    times_with_bailout, counts_with_bailout = collect_counts_and_times("shift_with_bailout", n)
    
    print(f"{'-' * 30}\n-- x = torch.ones(({n},)) ------\n{'-' * 30}\n")
    print("shift(x)")
    print("\n".join(repr(times).splitlines()[2:]), "\n")
    print("\n".join(trim_count_repr(counts).splitlines()[6:]))
    print(f"\n{'-' * 80}\n")
    print("shift_with_bailout(x)")
    print("\n".join(repr(times_with_bailout).splitlines()[2:]), "\n")
    print("\n".join(trim_count_repr(counts_with_bailout).splitlines()[6:]), "\n" * 3)

    lines = []
    for c, fn in delta(counts, counts_with_bailout):
        line = f"{c:>8} {fn}"
        lines.append(line[:110] + "..." if len(line) > 110 else line)

    print("Instruction deltas:")
    lines = lines if len(lines) < 16 else lines[:8] + ["..."] + lines[-8:]
    for l in lines:
        print(l)

    difference = (
        counts_with_bailout.counts(include_lookdict_unicode=False) / 
        counts.counts(include_lookdict_unicode=False))
    print(f"\n\n{(difference - 1) * 100:5.1f}% difference in instruction count.")

######################################################################
# When n=0 fast path is triggered
#

render_fast_path_effect(n=0)

# ------------------------------
# -- x = torch.ones((0,)) ------
# ------------------------------
#
# shift(x)
#   Median: 2.29 us
#   IQR:    0.10 us (2.24 to 2.33)
#   870 measurements, 10000 runs per measurement, 1 thread 
#
#                          All          Noisy symbols removed
#   Instructions:       977221                     963223
#   Baseline:             4610                       4180
#
# --------------------------------------------------------------------------------
#
# shift_with_bailout(x)
#   Median: 1.48 us
#   IQR:    0.09 us (1.45 to 1.54)
#   1259 measurements, 10000 runs per measurement, 1 thread 
#
#                          All          Noisy symbols removed
#   Instructions:       672488                     658523
#   Baseline:             4610                       4180 
#
#
#
# Instruction deltas:
#     9600 build/../aten/src/ATen/core/dispatch/Dispatcher.h:at::Tensor c10::Dispatcher::callWithDispatchKey<at:...
#     3300 build/aten/src/ATen/BackendSelectRegister.cpp:at::(anonymous namespace)::empty_memory_format(c10::Arr...
#     3300 build/aten/src/ATen/BackendSelectRegister.cpp:at::(anonymous namespace)::empty_memory_format(c10::Arr...
#     3300 build/../aten/src/ATen/core/boxing/KernelFunction_impl.h:at::Tensor c10::Dispatcher::callWithDispatch...
#     3200 build/../c10/core/TensorOptions.h:c10::TensorOptions::computeDispatchKey() const
#     3200 build/../aten/src/ATen/core/dispatch/Dispatcher.h:at::Tensor c10::Dispatcher::callWithDispatchKey<at:...
#     3100 build/../torch/csrc/autograd/generated/VariableType_4.cpp:torch::autograd::VariableType::(anonymous n...
#     2700 build/../c10/util/Optional.h:at::Tensor c10::Dispatcher::callWithDispatchKey<at::Tensor, c10::ArrayRe...
# ...
#    -7200 build/../c10/core/Device.h:c10::Device::validate()
#    -7200 build/../aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:c10::impl::wrap_kernel_func...
#    -7500 build/../aten/src/ATen/core/boxing/KernelFunction_impl.h:at::Tensor c10::Dispatcher::callWithDispatch...
#   -10200 build/../c10/util/Optional.h:c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionI...
#   -12000 build/../aten/src/ATen/core/dispatch/Dispatcher.h:at::Tensor c10::Dispatcher::callWithDispatchKey<at:...
#   -12300 build/../c10/util/Optional.h:at::Tensor c10::Dispatcher::callWithDispatchKey<at::Tensor, c10::ArrayRe...
#   -16000 ???:_dl_update_slotinfo
#   -30400 build/../c10/core/TensorOptions.h:c10::TensorOptions::merge_in(c10::TensorOptions) const
#
#
# -31.6% difference in instruction count.

######################################################################
# And for the normal path
#

render_fast_path_effect(n=1)

# ------------------------------
# -- x = torch.ones((1,)) ------
# ------------------------------
#
# shift(x)
#   Median: 2.65 us
#   IQR:    0.09 us (2.61 to 2.70)
#   739 measurements, 10000 runs per measurement, 1 thread 
#
#                          All          Noisy symbols removed
#   Instructions:      1111799                    1097920
#   Baseline:             4610                       4180
#
# --------------------------------------------------------------------------------
#
# shift_with_bailout(x)
#   Median: 2.71 us
#   IQR:    0.10 us (2.67 to 2.76)
#   723 measurements, 10000 runs per measurement, 1 thread 
#
#                          All          Noisy symbols removed
#   Instructions:      1118075                    1099974
#   Baseline:             4610                       4180 
#
#
#
# Instruction deltas:
#     1300 shift_impl_v2/main.cpp:shift_with_bailout(at::Tensor const&)
#      800 torch/include/ATen/core/TensorBody.h:shift_with_bailout(at::Tensor const&)
#      400 build/../c10/core/TensorImpl.h:c10::TensorImpl::numel() const
#       51 ???:_int_malloc
#      -41 ???:_int_memalign
#      -56 ???:_int_free
#     -200 build/../c10/core/Allocator.h:c10::TensorImpl::data() const
#     -200 build/../aten/src/TH/THAllocator.cpp:getTHDefaultAllocator()
#
#
#   0.2% difference in instruction count.

######################################################################
# There is a difference, but with wall clock it's difficult to
# differentiate minor regressions from noise. Instruction counts, by
# contrast, are deterministic and highlight not only how much more work
# is being done but also where it is being done.
#

######################################################################
# Congratulations! You have successfully benchmarked code in PyTorch.
# 
# Learn More
# ----------
# 
# Take a look at the following tutorial to learn how to profile your model.
# 
# - `PyTorch Profiler <https://https://pytorch.org/tutorials/recipes/recipes/profiler.html>`__
#
