Starter Code for the Matrix Multiplication assignment
Original code provided by Jim Demmel
http://www.cs.berkeley.edu/~knight/cs267/hw1.html
with some modifications by Scott B. Baden at UC San Diego
with some modifications by Bryan Chin at UC San Diego

### dgemm-blocked.c configuration
- `__GOTO__`: enable goto blocked multiplication
    - `__AVX256__`, `__AVX512__`: choose the size of SIMD; otherwise, naive is chosen
    - `NC`, `KC`, `MC`, `MR`, `NR`: block size configuration
        - `KC` \* `NC` for L3 cache
        - `MC` \* `KC` for L2 cache
        - `KC` \* `NR` for L1 cache
- `__LEVEL__`: enable 3-level blocked multiplication
    - `block_size_l2`, `block_size_l1`, `block_size_sm`: block size configuration
        - `block_size_l2` \* `block_size_l2` for L2 cache
        - `block_size_l1` \* `block_size_l1` for L1 cache
        - `block_size_sm` \* `block_size_sm` for register
    - `__AVX256__`: enable SIMD in the most inner block
- simple blocked method
    - `BLOCK_SIZE`: the block size of inner block
    - `TRANSPOSE`: transpose matrix `B` or not
### gcc optimization flags
- `-Ofast`: for global optimization
- `-ffast-math`: faster math computation
- `-funroll-loops`, `-funroll-all-loops`: loog unrolling
- `-mfpmath=sse`: auto generate floating-point arithmetic with `SSE`
- `-msse`, `-mavx`, `-mavx512f`, `-mfma`: enable different SIMD
