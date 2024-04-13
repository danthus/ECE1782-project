# ece1782 project, block matching algorithm
main_alternateive.cu, an alternative structure, not used anymore.
main_basic1.cu, contains the basic version of GPU implementation of block matching algorithm
main_lessdiver2.cu, applied less divergence optimization
main_reduction3.cu, replaced linear minimum finding to reduction minimum finding
main_reduction4.cu, unroll the last few loops in the reduction loop
main_shrd5.cu, move the current block in the SAD summation process to shared memory.

usage: ./a.out ***.yuv <width> <height> <block size> <search range>
examples:
  nvcc main_xxxx.cu
  ./a.out test.yuv 1920 1080 8 8 
  ./a.out test_720p.yuv 1280 720 4 16
  ./a.out CIF.yuv 352 288 16 32 

The CPU folder contains the optimized CPU codes:

