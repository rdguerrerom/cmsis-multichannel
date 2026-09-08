# Multi-channel filtering throughput in CMSIS-DSP

Reproduction harness for a measurement on [CMSIS-DSP](https://github.com/ARM-software/CMSIS-DSP):
filtering C independent channels through the same cascade is substantially
faster when the channels are processed together than when the library's
one-signal-per-call API is invoked C times — for kernels whose *within-signal*
parallelism is blocked by a loop-carried dependency.

It does not always help. `arm_fir_f32` is included precisely because the same
technique **loses** there.

## Build and run

```sh
git clone --depth 1 https://github.com/ARM-software/CMSIS-DSP
CM=CMSIS-DSP

clang -O3 -std=c11 -I $CM/Include -I $CM/PrivateInclude -DARM_MATH_NEON \
    -o bench_iir_lattice bench_iir_lattice_multichannel.c \
    $CM/Source/FilteringFunctions/arm_iir_lattice_f32.c \
    $CM/Source/FilteringFunctions/arm_iir_lattice_init_f32.c -lm

./bench_iir_lattice 64 16 4096      # channels, numStages, blockSize
```

The other two build the same way against `arm_fir_f32.c` / `arm_fir_init_f32.c`
and `arm_fir_lattice_f32.c` / `arm_fir_lattice_init_f32.c`.

Each benchmark **compares its output against CMSIS-DSP's** and refuses to print
a timing unless they agree. `max_abs_diff` is in the JSON it prints.

## What it measures

Both sides filter the same C channels with the same coefficients:

* **CMSIS path** — one `arm_*` instance per channel, called C times. Where the
  benchmark can, it also times the channel-major layout the library prefers, so
  the comparison is not charging it for a transpose it would not do.
* **Batched path** — one pass with the channel index innermost, so each stage
  update is a vector operation across channels while the time dependency stays
  on the outer loop.

Footprint for both is reported in the same JSON, in bytes, for C channels
streaming continuously.

## Results on one machine

Apple M5, Homebrew clang, `-O3 -DARM_MATH_NEON`, f32, against CMSIS-DSP
`c0c8640`. Minimum of 15 runs. Absolute timings on this host vary by up to ~2×
between sessions, so only ratios measured within a run are meaningful.

`arm_iir_lattice_f32`, 64 channels, blockSize 4096 — ns per sample per channel:

| numStages | CMSIS | batched | | max_abs_diff |
|--:|--:|--:|--:|--:|
| 4 | 7.24 | 0.73 | 9.8× | `0.0e+00` |
| 8 | 8.98 | 1.21 | 7.3× | `0.0e+00` |
| 16 | 13.48 | 2.24 | 5.7× | `0.0e+00` |
| 32 | 24.92 | 4.15 | 5.2× | `0.0e+00` |

At 32 channels, 16 stages, blockSize 32: **3.1×**.

`arm_fir_lattice_f32`, 64 channels, 16 stages: **4.9×**, `0.0e+00`.

**`arm_fir_f32`, 8 channels, 32 taps: 0.50× — half the speed.** A FIR output is
an independent dot product over taps, so its parallelism is already reachable
within one channel and the existing NEON path exploits it. The technique also
loses below roughly 8 channels on every kernel here.

## Exactness

The lattice kernels are **bit-identical** to CMSIS-DSP — `0.0e+00` maximum
absolute difference — because the arithmetic per channel is unchanged and only
the loop order differs.

The biquad cascade is not: it agrees to about `4.5e-07`, which is f32 rounding.
That is a weaker claim and is stated separately for that reason.

## Footprint

The batched state is `(numStages+1) × C` floats and does not depend on
blockSize. `arm_iir_lattice_init_f32` requires `pState` of
`numStages + blockSize` per instance, which a streaming C-channel system pays C
times.

Against the best a careful integrator could reach instead — `numStages` per
channel plus one shared `blockSize` scratch — the two are equal at blockSize 32
and the batched form is smaller above it. At a 16-sample block the batched form
uses 8.6% more.

## Caveats

* One machine, one compiler, Cortex-A/NEON. No M-profile hardware was
  available, so nothing here says how it behaves under Helium, where the
  tradeoff may differ.
* f32 only. Q15/Q31 are untested.
* The benchmarks are single-threaded and measure steady-state block processing.
