#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>

__global__ void block_matching(int* ref_frame, int* curr_frame, int width, int height, int blk_size, int srch_range)
{

}

// usage: main.cu 
int main( int argc, char *argv[])
{
    if( argc != 3) {
        printf( "Error: wrong number of args\n");
        exit(1);
    }
}