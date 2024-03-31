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

double getTimeStamp() {
    struct timeval tv;
    gettimeofday( &tv, NULL );
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

__global__ void block_matching(uint8_t* ref_frame, uint8_t* curr_frame, int* mv, int width, int height, int blk_size, int srch_range)
{
    int macro_x = blockIdx.x * blockDim.x;
    int macro_y = blockIdx.y * blockDim.y;

    int offset_x = threadIdx.x - srch_range;
    int offset_y = threadIdx.y - srch_range;
    int SAD = 0;

    int ref_x = (int)(blockIdx.x*blockDim.x)+offset_x;
    int ref_y = (int)(blockIdx.y*blockDim.y)+offset_y;

    // if(blockIdx.x ==22 && blockIdx.y==7)
    //     printf("ref: %d, %d \n", ref_y, ref_x);

    if(ref_x<0 || ref_x>width-BLK_SIZE || ref_y<0 || ref_y>height-BLK_SIZE)
    {
        ref_x = macro_x;
        ref_y = macro_y;
        SAD = 999999; // give large value for outsiders
    }

    __shared__ int SAD_list[100];
    __shared__ int mv_list[200];

    for(int i=0; i<BLK_SIZE; i++)
    {
        for(int j=0; j<BLK_SIZE; j++)
        {
            SAD += abs((int)ref_frame[ref_y*width+ref_x + i*width+j] - (int)curr_frame[macro_y*width+macro_x + i*width+j]);
        }
    }

    SAD_list[threadIdx.y*2*srch_range + threadIdx.x] = SAD;
    mv_list[threadIdx.y*2*srch_range*2 + threadIdx.x*2] = offset_y;
    mv_list[threadIdx.y*2*srch_range*2 + threadIdx.x*2+1] = offset_x;
    __syncthreads();

    // if(blockIdx.x ==43 && blockIdx.y==0)
    //     printf("ref: %d \n", SAD);

    if(threadIdx.x == 0 && threadIdx.y == 0)
    {
        // if(blockIdx.x ==22 && blockIdx.y==7)
        //     printf("%d %d %d \n", SAD_list[0], mv_list[0], mv_list[1]);

        int min_SAD = SAD_list[0];
        int best_y = -4;
        int best_x = -4; 
        for(int k=1; k<2*srch_range*2*srch_range; k++)
        {
            // if(blockIdx.x ==22 && blockIdx.y==7)
            //     printf(" minSAD %d %d %d \n", min_SAD, best_y, best_x);
            if(min_SAD >= SAD_list[k])
            {
                min_SAD = SAD_list[k];
                best_y = mv_list[2*k];
                best_x = mv_list[2*k+1];
            }
            // if(blockIdx.x ==22 && blockIdx.y==7)
            //     printf(" %d %d %d \n", SAD_list[k], mv_list[2*k], mv_list[2*k+1]);
        }
        // if(blockIdx.x ==22 && blockIdx.y==7)
        //     printf("\n best y %d, best x %d \n", best_y, best_x);
        mv[blockIdx.y*gridDim.x*4 + blockIdx.x*4] = blockIdx.y * blockDim.y;
        mv[blockIdx.y*gridDim.x*4 + blockIdx.x*4 + 1] = blockIdx.x * blockDim.x;
        mv[blockIdx.y*gridDim.x*4 + blockIdx.x*4 + 2] = best_y;
        mv[blockIdx.y*gridDim.x*4 + blockIdx.x*4 + 3] = best_x;
    }

}

void host_block_matching(uint8_t* ref_frame, uint8_t* curr_frame, int* mv, int width, int height, int blk_size, int srch_range)
{
    int SAD, minSAD, best_x, best_y;
    for(int macro_y=0; macro_y < height; macro_y+=BLK_SIZE)
    {
        for(int macro_x=0; macro_x < width; macro_x+=BLK_SIZE)
        {
            minSAD = 99999;
            for(int ref_y=macro_y-srch_range; ref_y < macro_y+srch_range; ref_y++)
            {
                for(int ref_x=macro_x-srch_range; ref_x < macro_x+srch_range; ref_x++)
                {
                    if(ref_x>=0 && ref_x<=width-BLK_SIZE && ref_y>=0 && ref_y<=height-BLK_SIZE)
                    {
                        SAD = 0;
                        for(int i=0; i<BLK_SIZE; i++)
                        {
                            for(int j=0; j<BLK_SIZE; j++)
                            {
                                SAD += abs((int)ref_frame[ref_y*width+ref_x + i*width+j] - (int)curr_frame[macro_y*width+macro_x + i*width+j]);
                            }
                        }
                        // if(macro_x == 96 && macro_y == 0)
                        //     printf("%d %d %d\n", SAD, ref_y, ref_x);

                        if(minSAD >= SAD)
                        {
                            minSAD = SAD;
                            best_x = ref_x - macro_x;
                            best_y = ref_y - macro_y;
                        }
                        // if(macro_x == 96 && macro_y == 0)
                        //     printf("%d %d \n", best_y, best_x);

                    }
                }
            }
            mv[(macro_y/BLK_SIZE)*(width/BLK_SIZE)*4 + (macro_x/BLK_SIZE)*4] = macro_y;
            mv[(macro_y/BLK_SIZE)*(width/BLK_SIZE)*4 + (macro_x/BLK_SIZE)*4 + 1] = macro_x;
            mv[(macro_y/BLK_SIZE)*(width/BLK_SIZE)*4 + (macro_x/BLK_SIZE)*4 + 2] = best_y;
            mv[(macro_y/BLK_SIZE)*(width/BLK_SIZE)*4 + (macro_x/BLK_SIZE)*4 + 3] = best_x;
        }
    }
}

bool compare_mv(int* mv1, int* mv2, size_t len)
{
    for(int i=0; i<len; i++)
    {
        if(mv1[i] != mv2[i])
        {
            printf("Error in mv: index %d, value1: %d, value2: %d \n", i, mv1[i], mv2[2]);
            return 0;
        }
    }
    return 1;
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
    int *h_d_mv; int *h_h_mv;
    gpuErrchk(cudaMallocHost((void **)&h_ref_frame, frame_bytes));
    gpuErrchk(cudaMallocHost((void **)&h_cur_frame, frame_bytes));
    gpuErrchk(cudaMallocHost((void **)&h_d_mv, vector_bytes));
    gpuErrchk(cudaMallocHost((void **)&h_h_mv, vector_bytes));

    int ret = read_next_frame(yuv_file, h_ref_frame, frame_bytes);
    printf("ret = %d \n", ret);
    read_next_frame(yuv_file, h_cur_frame, frame_bytes);

    // for(int i=0; i < 20; i++)
    //     printf("%hhu ", h_cur_frame[i]);
    // printf("\n");

    dim3 dimGrid((WIDTH+BLK_SIZE-1)/BLK_SIZE, (HEIGHT+BLK_SIZE-1)/BLK_SIZE);
    dim3 dimBlock(BLK_SIZE, BLK_SIZE);

    // Device memory allocation
    double start_time=getTimeStamp();

    uint8_t *d_ref_frame, *d_cur_frame;
    int *d_mv;
    cudaMalloc((void **)&d_ref_frame, frame_bytes);
    cudaMalloc((void **)&d_cur_frame, frame_bytes);
    cudaMalloc((void **)&d_mv, vector_bytes);

    cudaMemcpy(d_ref_frame, h_ref_frame, frame_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cur_frame, h_cur_frame, frame_bytes, cudaMemcpyHostToDevice);

    block_matching<<<dimGrid, dimBlock>>>(d_ref_frame, d_cur_frame, d_mv, WIDTH, HEIGHT, BLK_SIZE, SRC_range);

    cudaDeviceSynchronize();

    cudaMemcpy(h_d_mv, d_mv, vector_bytes, cudaMemcpyDeviceToHost);
    double end_time=getTimeStamp();

    // do compression using h_mv

    // CPU GPU check ////////////
    host_block_matching(h_ref_frame, h_cur_frame, h_h_mv, WIDTH, HEIGHT, BLK_SIZE, SRC_range);
    if(compare_mv(h_h_mv, h_d_mv, vector_size))
        printf("CPU GPU check success\n");
    else
        printf("CPU GPU check failed\n");
        
    int total_time_ms =(int)ceil((end_time-start_time)*1000);
    printf("time: %d ms\n", total_time_ms);
    /////////////////////////////

    for(int i=1314; i<1314+32; i++)
        printf("%d ", h_d_mv[i]);
    printf("\n");

    for(int i=1314; i<1314+32; i++)
        printf("%d ", h_h_mv[i]);
    printf("\n");


    cudaFreeHost(h_ref_frame);
    cudaFreeHost(h_cur_frame);
    cudaFreeHost(h_d_mv);
    cudaFreeHost(h_h_mv);
    cudaFree(d_ref_frame);
    cudaFree(d_cur_frame);
    cudaFree(d_mv);
    fclose(yuv_file);
    return 0;
}