# ece1782 project, block matching algorithm
main_alternateive.cu, an alternative structure, not used anymore.

main_basic1.cu, contains the basic version of GPU implementation of block matching algorithm

main_lessdiver2.cu, applied less divergence optimization

main_reduction3.cu, replaced linear minimum finding to reduction minimum finding

main_reduction4.cu, unroll the last few loops in the reduction loop

main_shrd5.cu, move the current block in the SAD summation process to shared memory.

test.yuv, contains 10 frames of a 1920x1080 resolution video

test_720p.yuv, contains 10 frames of a 1280x720 resolution video

CIF.yuv, contains 10 frames of a 352x288 resolution video

usage: 

    ./a.out <yuv file> <width> <height> <block size> <search range>

examples:

    nvcc main_xxxx.cu
    
    ./a.out test.yuv 1920 1080 8 8 
    
    ./a.out test_720p.yuv 1280 720 4 16
    
    ./a.out CIF.yuv 352 288 16 32 

The CPU folder contains the optimized CPU codes:

cpu_naive1.cu, contains the naive implementation of block matching CPU code

cpu_earlystop2.cu, contains early stop optimization CPU code

cpu_parallel3.cpp, contains the parallel optimization using openMP library CPU code

cpu_unroll4.cpp, contains the unroll optimization by completely unrolling the inner-most loop

usage for the .cu files are the same as the main_xxx.cu file.

usage for the .cpp file are:

    ./a.out <yuv file> <width> <height> <block size> <search range>

examples:

    g++ -fopenmp cpu_parallel3.cpp
    
    g++ -fopenmp cpu_unroll4.cpp

    ./a.out ../test.yuv 1920 1080 8 32
    

