# Literal unary CUDA pulse-array primitives

Tested on an NVIDIA GTX 1660 SUPER, compute capability 7.5, driver 615.71.09 and CUDA compiler 12.8.93. No install required there.

```sh
nvcc -std=c++17 -O2 -arch=sm_75 pulses.cu -o pulses
./pulses 419071
./pulses 457201
nvcc -std=c++17 -O2 -arch=sm_75 --ptx pulses.cu -o device.ptx
```

Each occupied pulse is a separate byte slot, not an integer magnitude or a packed popcount. A separate categorical tag gives the sign. Addition consumes matching pulses for cancellation then copies survivors. The QK kernel shown is a one-coordinate signed product, not a full vector dot or multi-head Transformer; it materializes one output slot per possible pulse pair. Hardmax compares magnitude by matching occupied pulse pairs until one side is exhausted, combines the result with sign categories, and preserves first-key ties. Device code uses integer cursors for array addressing and kernel launch topology, boolean tests and selection labels, not decoded scalar numeric values for sums or products. Generated PTX contained three device entries and no floating-point instructions or popcount; host timing uses float/double separately.

Each fresh process compared 4,096 signed additions (including all 33 × 33 ordered input pairs over −16 through 16), 32,768 signed scalar QK products and 4,096 hardest-key selections to an independent literal string-tally CPU reference. QK is checked at each allocated pulse position, by sign and by decoded string. Both seeds passed exactly. This initial CPU oracle was for correctness only; the matched CPU comparison is reported below.

The timed region includes three GPU kernels per repetition; 64 repetitions took 436.07 ms and 427.35 ms, or 6.81 and 6.68 ms per 4,096-row triple. The allocation-through-transfer host intervals of 445.47 and 436.45 ms include those timed repetitions and warmup, not compilation or CPU oracle verification. Own allocations total 9,326,592 bytes (8.89 MiB) and the GPU's post-context free-memory counter dropped 12 MiB. After the test, device usage returned to 1 MiB, 0% and no compute PID. This is primitive parity and profiling only, not CUDA model training or a claim of speedup.

## Matched performance follow-up

Compile `nvcc -std=c++17 -O2 -arch=sm_75 -Xcompiler=-O3,-march=native bench.cu -o bench` and run `./bench 419071` and `./bench 457201`. The benchmark imports the identical CUDA kernels from `pulses.cu`. The CPU counterpart uses the same unpacked pulse buffers and hardmax tie rules; its pointwise row copy/zero is optimized by the C++ compiler. Host work uses one pinned Ryzen 5 5600 core. Byte-for-byte parity is checked after every arm. We alternated starting arm across nine six-repetition rounds per fresh seed. Device-resident timing uses CUDA events; transfer-inclusive timing includes input upload and output download via pageable host buffers. Allocations, context startup and parity checks are excluded from all timed arms.

For each 4,096-row workload (eight candidates per row), the combined speed ranking was:

| Arm | Median ms | Speed index (fastest = 100) |
| --- | ---: | ---: |
| Literal pulse CPU | 3.384 | 100 |
| GTX 1660 SUPER, device-resident | 6.666 | 50.77 |
| GTX 1660 SUPER, with transfers | 8.527 | 39.69 |

This run therefore did not show a GPU speedup: CUDA was 1.97× slower resident and 2.52× slower with transfers. The design materializes every 256-byte QK pulse footprint, and the selected design scans those footprints serially within each GPU thread; it is a representation-preserving reference, not the best possible CUDA schedule. No broader hardware conclusion follows. Raw matched timings are in `bench.419071.json` and `bench.457201.json`; run `python3 -B bench_report.py` to reproduce aggregate statistics.
