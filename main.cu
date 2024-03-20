#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>

/*You can use the following for any CUDA function that returns cudaError_t type*/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code == cudaSuccess) return;

    fprintf(stderr,"Error: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
}

__global__ void block_matching(int* ref_frame, int* curr_frame, int width, int height, int blk_size, int srch_range)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    
}

// usage: main.cu <video file> <WIDTH> <HEIGHT> <BLK_size> <search range>
int main( int argc, char *argv[])
{
    // if( argc != 3) {
    //     printf( "Error: wrong number of args\n");
    //     exit(1);
    // }
    
    int WIDTH = atoi(argv[2]);
    int HEIGHT = atoi(argv[3]);
    int BLK_SIZE = atoi(argv[4]);
    int SRC_range = atoi(argv[5]);

    // video file preprocess

    // !!!no padding right now, need to handle it!!!
    size_t pixels = WIDTH*HEIGHT;
    size_t pixel_bytes = pixels*sizeof(uint8_t);
    size_t vector_size = 4*(pixels/BLK_SIZE/BLK_SIZE); // format: y x mv_y mv_x ...
    size_t vector_bytes = vector_size*sizeof(int);

    // Host memory allocation
    uint8_t *h_ref_frame, *h_cur_frame;
    int *h_mv;
    gpuErrchk(cudaMallocHost((void **)&h_ref_frame, bytes));
    gpuErrchk(cudaMallocHost((void **)&h_cur_frame, bytes));
    gpuErrchk(cudaMallocHost((void **)&h_mv, vector_bytes));


    dim3 dimGrid((WIDTH+BLK_SIZE-1)/BLK_SIZE, (HEIGHT+BLK_SIZE-1)/BLK_SIZE);
    dim3 dimBlock(BLK_SIZE, BLK_SIZE);

    // Device memory allocation
    uint8_t *d_ref_frame, *d_cur_frame;
    int *d_mv;
    cudaMalloc((void **)&d_ref_frame, bytes);
    cudaMalloc((void **)&d_cur_frame, bytes);
    cudaMalloc((void **)&d_mv, vector_bytes);

    cudaMemcpy(d_ref_frame, h_ref_frame, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cur_frame, h_cur_frame, bytes, cudaMemcpyHostToDevice);

    block_matching<<<dimGrid, dimBlock>>>(d_ref_frame, d_cur_frame, BLK_SIZE, SRC_range);

    cudaDeviceSynchronize();

    cudaMemcpy(h_mv, d_mv, vector_bytes, cudaMemcpyDeviceToHost);

    // do compression using h_mv



    cudaFreeHost(h_ref_frame);
    cudaFreeHost(h_cur_frame);
    cudaFreeHost(h_mv);
    cudaFree(d_ref_frame);
    cudaFree(d_cur_frame);
    cudaFree(d_mv);
    return 0;
}