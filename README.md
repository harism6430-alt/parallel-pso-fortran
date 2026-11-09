# Particle Swarm Optimization in Fortran with OpenMP

A particle swarm optimizer written in modern Fortran and parallelized with OpenMP. I wrote it for the course *Introduction to Parallel and Scientific Computing* in the Materials Science and Simulation (MSS) master's program at Ruhr University Bochum, summer semester 2026.

The program does three things:

1. Tests the optimizer on seven analytical benchmark functions with known minima.
2. Studies how the split between self trust (C1) and group trust (C2) changes the result, using point charges on a unit disk as the test problem.
3. Places up to 64 repelling charges (128 variables) and measures how much OpenMP speeds up a single run.

## Build and run

You need `gfortran` with OpenMP support and `make`.

```
make
OMP_NUM_THREADS=16 ./pso_run
```

`make run-16` does the same thing. The program caps the thread count at 16, or at the number of cores if the machine has fewer. A full run on 16 threads takes about 6 minutes (roughly 1 s for task 1, 90 s for task 2, 270 s for task 3). It writes CSV files into the working directory.

To make the plots from the CSV files (needs numpy, pandas and matplotlib):

```
python plot_results.py
```

## What the optimizer does

Every particle updates its velocity and position like this:

```
v = w*v + C1*r1*(pbest - x) + C2*r2*(gbest - x)
x = x + v*dt
```

r1 and r2 are fresh random numbers in [0, 1] for every particle, dimension and iteration. A few additions to the plain version:

- The inertia weight w falls linearly from 0.9 to 0.4 over the run.
- Velocity is clamped, to 10% of the search range for the benchmarks and 20% for the charges. Initial velocities are random, not zero, because with C1 = 0 a zero start leaves the swarm standing still.
- Step size adapts. If the relative change of the best value stays below 1e-8 for 20 iterations in a row, dt is halved, up to 12 times.
- For charges, any particle that ends up outside the unit disk is projected back onto the boundary.

## Task 1: benchmark functions

Each function is two dimensional and was run 20 times with different seeds. "Best" is the best of the 20 runs.

| Function | Best of 20 | Mean of 20 | Std dev | True minimum |
|---|---|---|---|---|
| Radial | 0.00000000 | 0.00000000 | 0 | 0 |
| McCormick | -1.91322295 | -1.91322295 | 0 | -1.91322295 |
| Himmelblau | 0.00000000 | 0.00000000 | 0 | 0 |
| 3-Hump Camel | 0.00000000 | 0.00000000 | 0 | 0 |
| 6-Hump Camel | -1.03162845 | -1.03162845 | 0 | -1.03162845 |
| Eggholder | -959.64066142 | -916.03 | 65.27 | -959.64070000 |
| Rastrigin | -20.00000000 | -20.00000000 | 0 | -20 |

Six functions hit the true minimum in all 20 runs. Eggholder is the hard one: the best run is within 0.000004% of the true minimum, but the average run is not, and the spread between runs is large. For the three functions whose true minimum is 0 the program reports an absolute error instead of a relative one.

Settings: most functions use 100 particles, 500 iterations and C1 = C2 = 1.496. McCormick uses 150 particles, 1200 iterations, C1 = C2 = 0.5 and the search range widened to [-2, 4] so that the minimum at (-0.547, -1.547) lies inside it. Eggholder uses 150 particles, 800 iterations, C1 = 0.3 and C2 = 0.7.

## Task 2: C1 and C2

C1 + C2 is fixed at 1 and C1 goes from 0.0 to 1.0 in steps of 0.1. For each of the 11 settings and each system of 1 to 6 charges, the optimizer ran 100 times. The runs are independent, so they are spread over threads with `omp parallel do` (task farming). Seeds are fixed, so this part gives the same numbers on every run.

Mean final energy for the largest systems:

| Charges | C1 = 0.0 | C1 = 1.0 (no group trust) | Difference |
|---|---|---|---|
| 4 | 3.8285 | 3.9596 | 3.4% |
| 5 | 6.8821 | 7.3564 | 6.9% |
| 6 | 10.9739 | 11.9093 | 8.6% |

What the numbers show:

- Pure self trust (C1 = 1) is clearly worse, and the gap grows with the number of charges.
- Between C1 = 0 and about C1 = 0.8 the final energy hardly changes. The sharp rise only starts around 0.9. So "group trust wins" is true against pure self trust, but not very sensitive to the exact value below that.
- For 6 charges the lowest mean (10.9652) was at C1 = 0.2, not 0.0. The difference is inside the run to run noise.
- Self trust converges in fewer iterations, but to a worse energy.

## Task 3: charges on a unit disk

N charges sit inside the unit disk and the energy is the sum of 1/distance over all pairs. Each configuration was run 20 times.

| Charges | Variables | Particles | Iterations | C1 / C2 | Best energy | Mean energy |
|---|---|---|---|---|---|---|
| 6 | 12 | 150 | 800 | 0.1 / 0.9 | 10.9642 | 10.9649 |
| 16 | 32 | 200 | 1000 | 0.4 / 0.6 | 116.58 | 117.4 |
| 32 | 64 | 300 | 1500 | 0.4 / 0.6 | 554.1 | 556.4 |
| 64 | 128 | 400 | 2000 | 0.4 / 0.6 | 2490.0 | 2497.6 |

For 6 charges the best energy matches the closed form for a regular hexagon on the rim, 10.9641. For more charges there is no simple reference (some charges move inside the disk), so I only report best and mean over the 20 runs. I did not check these against an independent global optimizer. The 16, 32 and 64 charge numbers change slightly from run to run, because OpenMP hands out particles dynamically and so the random streams are not tied to particles. The values above are from one run.

Particle swarm settings (particles, iterations, C1 and C2) were picked by trial, with no systematic sweep.

### Speedup

The speedup test runs the same 64 charge problem (300 particles, 300 iterations, C1 = C2 = 0.5, same seed) once serially and once with the particle loop split over 16 threads:

| | Wall clock time |
|---|---|
| 1 thread | 0.886 s |
| 16 threads | 0.069 s |
| Speedup | 12.9 times |

It is one measurement of a short run, not a scaling study, and I did not test 2, 4 or 8 threads. The two runs end at slightly different energies (2577.9 and 2570.9) for the same reason as above. Time is taken with `system_clock`, because `cpu_time` adds up the time of all threads and would hide the speedup.

The parallel part is the particle loop (`schedule(dynamic, 10)`). The global best is read only while particles move and is updated in a `single` block after each iteration.

## A random number bug I fixed

My first version used a function `srnd(seed)` that remembered its seed. After the first call in each thread it ignored the seed it was given and just kept advancing one sequence. The results were statistically fine, but the same seed no longer gave the same run when the thread count changed. The fix seeds the generator once per particle at start-up and once per thread at the start of the parallel region, and draws the numbers with `rnd()` from a thread private state. `srnd` is still in `functions_rnd_module.f90` but nothing uses it any more.

## Files

| File | Content |
|---|---|
| `main.f90` | runs the three tasks in order, sets the thread count |
| `pso_core.f90` | the optimizer and its parameter and result types |
| `functions.f90` | the seven benchmark functions, the charge energy |
| `functions_rnd_module.f90` | random number generator |
| `data_kind.f90` | real and integer kinds |
| `task1_benchmark.f90` | task 1 |
| `task2_sensitivity.f90` | task 2 |
| `task3_charge.f90` | task 3 and the speedup test |
| `plot_results.py` | makes the figures from the CSV files |
| `Makefile` | build, run and clean targets (`make help` lists them) |
| `results/` | CSV files and figures from my run |

Author: Muhammad Haris, M.Sc. Materials Science and Simulation, Ruhr University Bochum.
