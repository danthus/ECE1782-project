#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define BLK_SIZE 8

/*You can use the following for any CUDA function that returns cudaError_t type*/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code == cudaSuccess) return;

    fprintf(stderr,"Error: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
}

__global__ void block_matching(uint8_t* ref_frame, uint8_t* curr_frame, int* mv, int width, int height, int blk_size, int srch_range)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    int offset_x = threadIdx.x - srch_range;
    int offset_y = threadIdx.y - srch_range;
    int curr_idx = iy*width + ix;
    int ref_idx = curr_idx + offset_y*width + offset_x;
    int SAD = 0;
    if(ref_idx < 0 || ref_idx > width*height - (BLK_SIZE*width+BLK_SIZE))
    {
        ref_idx = curr_idx; 
        SAD = 999999; // give large value for outsiders
    }

    __shared__ int SAD_list[100];
    __shared__ int mv_list[200];

    for(int i=0; i<BLK_SIZE; i++)
    {
        for(int j=0; j<BLK_SIZE; j++)
        {
            SAD += abs((int)ref_frame[ref_idx+i*width+j] - (int)curr_frame[curr_idx+i*width+j]);
        }
    }

    SAD_list[threadIdx.y*2*srch_range + threadIdx.x] = SAD;
    mv_list[threadIdx.y*2*srch_range*2 + threadIdx.x*2] = offset_y;
    mv_list[threadIdx.y*2*srch_range*2 + threadIdx.x*2+1] = offset_x;
    __syncthreads();

    if(threadIdx.x == 0 && threadIdx.y == 0)
    {
        int min_SAD = SAD_list[0];
        int best_y = 0;
        int best_x = 0; 
        for(int k=1; k<2*srch_range*2*srch_range; k++)
        {
            if(min_SAD >= SAD_list[k])
            {
                min_SAD = SAD_list[k];
                best_y = mv_list[k];
                best_x = mv_list[k+1];
            }
        }
        mv[blockIdx.y*gridDim.x*4 + blockIdx.x*4] = blockIdx.y * blockDim.y;
        mv[blockIdx.y*gridDim.x*4 + blockIdx.x*4 + 1] = blockIdx.x * blockDim.x;
        mv[blockIdx.y*gridDim.x*4 + blockIdx.x*4 + 2] = best_y;
        mv[blockIdx.y*gridDim.x*4 + blockIdx.x*4 + 3] = best_x;
    }

}

bool read_next_frame(FILE* yuv_file, uint8_t* frame_buffer, size_t frame_bytes)
{
    size_t result;
    result = fread(frame_buffer, 1, frame_bytes, yuv_file);
    fseek(yuv_file, frame_bytes/2, SEEK_CUR); // skip U, V components
    if (result == frame_bytes) 
        return 1;
    else
        return 0;
}

// usage: main.cu <video file> <WIDTH> <HEIGHT> <BLK_size> <search range>
int main( int argc, char *argv[])
{
    // if( argc != 3) {
    //     printf( "Error: wrong number of args\n");
    //     exit(1);
    // }
    
    int WIDTH = atoi(argv[1]);
    int HEIGHT = atoi(argv[2]);
    // int BLK_SIZE = atoi(argv[3]);
    int SRC_range = atoi(argv[3]);

    // video file preprocess

    // !!!no padding right now, need to handle it!!!
    size_t pixels = WIDTH*HEIGHT;
    size_t frame_bytes = pixels*sizeof(uint8_t);
    size_t vector_size = 4*(pixels/BLK_SIZE/BLK_SIZE); // format: y x mv_y mv_x ...
    size_t vector_bytes = vector_size*sizeof(int);

    // process frame
    FILE* yuv_file;
    yuv_file = fopen("CIF.yuv", "rb");
    if(yuv_file==NULL) {printf("Error: NULL file \n");exit(0);}
    rewind(yuv_file);

    // Host memory allocation
    uint8_t *h_ref_frame, *h_cur_frame;
    int *h_mv;
    gpuErrchk(cudaMallocHost((void **)&h_ref_frame, frame_bytes));
    gpuErrchk(cudaMallocHost((void **)&h_cur_frame, frame_bytes));
    gpuErrchk(cudaMallocHost((void **)&h_mv, vector_bytes));

    int ret = read_next_frame(yuv_file, h_ref_frame, frame_bytes);
    printf("ret = %d \n", ret);
    read_next_frame(yuv_file, h_cur_frame, frame_bytes);

    for(int i=0; i < 20; i++)
        printf("%hhu ", h_cur_frame[i]);
    printf("\n");

    dim3 dimGrid((WIDTH+BLK_SIZE-1)/BLK_SIZE, (HEIGHT+BLK_SIZE-1)/BLK_SIZE);
    dim3 dimBlock(BLK_SIZE, BLK_SIZE);

    // Device memory allocation
    uint8_t *d_ref_frame, *d_cur_frame;
    int *d_mv;
    cudaMalloc((void **)&d_ref_frame, frame_bytes);
    cudaMalloc((void **)&d_cur_frame, frame_bytes);
    cudaMalloc((void **)&d_mv, vector_bytes);

    cudaMemcpy(d_ref_frame, h_ref_frame, frame_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cur_frame, h_cur_frame, frame_bytes, cudaMemcpyHostToDevice);

    block_matching<<<dimGrid, dimBlock>>>(d_ref_frame, d_cur_frame, d_mv, WIDTH, HEIGHT, BLK_SIZE, SRC_range);

    cudaDeviceSynchronize();

    cudaMemcpy(h_mv, d_mv, vector_bytes, cudaMemcpyDeviceToHost);

    // do compression using h_mv

    for(int i=17*44*4; i<17*44*4+64; i++)
        printf("%d ", h_mv[i]);
    printf("\n");


    cudaFreeHost(h_ref_frame);
    cudaFreeHost(h_cur_frame);
    cudaFreeHost(h_mv);
    cudaFree(d_ref_frame);
    cudaFree(d_cur_frame);
    cudaFree(d_mv);
    fclose(yuv_file);
    return 0;
}