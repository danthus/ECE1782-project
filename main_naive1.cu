#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

// #define BLK_SIZE 8

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

__global__ void block_matching(uint8_t* ref_frame, uint8_t* curr_frame, int* mv, int width, int height, int BLK_SIZE, int srch_range)
{
    __shared__ int SAD_list[1024];
    __shared__ int mv_list[2048];

    int macro_x = blockIdx.x * blockDim.x;
    int macro_y = blockIdx.y * blockDim.y;
    int best_x=-99; int best_y=-99; int min_SAD=9999999;

    // if(blockIdx.x ==0 && blockIdx.y==0)
    //     printf("macro: %d %d %d\n", macro_x, blockIdx.y, blockIdx.x);

    // int sub_window_count = srch_range/BLK_SIZE;
    int sub_srch_range = BLK_SIZE/2;

    for(int sub_macro_y=macro_y-srch_range; sub_macro_y<macro_y+srch_range; sub_macro_y+=BLK_SIZE)
    {
        for(int sub_macro_x=macro_x-srch_range; sub_macro_x<macro_x+srch_range; sub_macro_x+=BLK_SIZE)
        {
            // int sub_macro_x = macro_x + sub_window_x*(srch_range + blk_size/2);
            // int sub_macro_y = macro_y + sub_window_y*(srch_range + blk_size/2);

            // int offset_x = threadIdx.x;
            // int offset_y = threadIdx.y;

            int ref_x = sub_macro_x + threadIdx.x;
            int ref_y = sub_macro_y + threadIdx.y;

            int SAD = 0;

            // if(blockIdx.x ==22 && blockIdx.y==7)
            //     printf("ref: %d, %d \n", ref_y, ref_x);

            // if(blockIdx.x ==119 && blockIdx.y==66)
            //     printf("macro: %d %d %d %d\n", macro_y, macro_x, ref_y, ref_x);

            for(int i=0; i<BLK_SIZE; i++)
            {
                for(int j=0; j<BLK_SIZE; j++)
                {
                    if(ref_x>=0 && ref_x<=width-BLK_SIZE && ref_y>=0 && ref_y<=height-BLK_SIZE)
                    {
                        SAD += abs((int)ref_frame[ref_y*width+ref_x + i*width+j] - (int)curr_frame[macro_y*width+macro_x + i*width+j]);
                    }
                    else
                    {
                        SAD = -1;
                    }
                }
            }

            // if(blockIdx.x ==119 && blockIdx.y==66)
            //     printf("macro: %d %d %d %d %d\n", macro_y, macro_x, ref_y, ref_x, SAD);

            SAD_list[threadIdx.y*2*sub_srch_range + threadIdx.x] = SAD;
            mv_list[threadIdx.y*2*sub_srch_range*2 + threadIdx.x*2] = ref_y - macro_y;
            mv_list[threadIdx.y*2*sub_srch_range*2 + threadIdx.x*2+1] = ref_x - macro_x;
            __syncthreads();

            if(threadIdx.x == 0 && threadIdx.y == 0)
            {
                // if(blockIdx.x ==39 && blockIdx.y==46)
                //     printf("%d %d %d \n", SAD_list[0], mv_list[0], mv_list[1]);

                // min_SAD = SAD_list[0];
                // best_y = mv_list[0];
                // best_x = mv_list[1]; 
                for(int k=0; k<2*sub_srch_range*2*sub_srch_range; k++)
                {
                    // if(blockIdx.x ==0 && blockIdx.y==0)
                    //     printf(" minSAD %d %d %d \n", min_SAD, best_y, best_x);
                    if(min_SAD > SAD_list[k] && SAD_list[k]!=-1)
                    {
                        min_SAD = SAD_list[k];
                        best_y = mv_list[2*k];
                        best_x = mv_list[2*k+1];
                        // if(blockIdx.x ==33 && blockIdx.y==13)
                        //     printf(" minSAD %d %d %d \n", min_SAD, best_y, best_x);
                    }
                    // if(blockIdx.x ==15 && blockIdx.y==3)
                    //     printf("%d %d %d %d %d %d\n", macro_y, macro_x, SAD_list[k], mv_list[2*k], mv_list[2*k+1], min_SAD);
                }
                // if(blockIdx.x ==22 && blockIdx.y==7)
                //     printf("\n best y %d, best x %d \n", best_y, best_x);
            }
            __syncthreads();
        }
    }

    if(threadIdx.x == 0 && threadIdx.y == 0)
    {
        size_t idx = blockIdx.y*gridDim.x*4 + blockIdx.x*4;
        // mv[blockIdx.y*gridDim.x*4 + blockIdx.x*4] = macro_y;
        // mv[blockIdx.y*gridDim.x*4 + blockIdx.x*4 + 1] = macro_x;
        // mv[blockIdx.y*gridDim.x*4 + blockIdx.x*4 + 2] = best_y;
        // mv[blockIdx.y*gridDim.x*4 + blockIdx.x*4 + 3] = best_x;
        mv[idx] = macro_y;
        mv[idx + 1] = macro_x;
        mv[idx + 2] = best_y;
        mv[idx + 3] = best_x;
    }
}

void host_block_matching(uint8_t* ref_frame, uint8_t* curr_frame, int* mv, int width, int height, int BLK_SIZE, int srch_range)
{
    int SAD, minSAD, best_x, best_y;
    for(int macro_y=0; macro_y < height; macro_y+=BLK_SIZE)
    {
        for(int macro_x=0; macro_x < width; macro_x+=BLK_SIZE)
        {
            minSAD = 9999999;
            for(int ref_y=macro_y-srch_range; ref_y < macro_y+srch_range; ref_y++)
            {
                for(int ref_x=macro_x-srch_range; ref_x < macro_x+srch_range; ref_x++)
                {
                    if(ref_x>=0 && ref_x<=width-BLK_SIZE && ref_y>=0 && ref_y<=height-BLK_SIZE)
                    {
                        SAD = 0;

                        for(int i=0; i<BLK_SIZE*BLK_SIZE; i++)
                        {
                                SAD += abs((int)ref_frame[ref_y*width+ref_x + (i/BLK_SIZE)*width+i%BLK_SIZE] - (int)curr_frame[macro_y*width+macro_x + (i/BLK_SIZE)*width+i%BLK_SIZE]);
                                // if(SAD >= minSAD)
                                //     break;
                        }

                        // if(macro_x == 480 && macro_y == 96)
                        //     printf("host: %d %d %d %d\n", SAD, ref_y-macro_y, ref_x-macro_x, minSAD);

                        if(minSAD > SAD)
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

bool compare_SAD(uint8_t* ref_frame, uint8_t* curr_frame, int macro_y, int macro_x, int mv1_y, int mv1_x, int mv2_y, int mv2_x, int width, int BLK_SIZE)
{
    int SAD1 = 0; int SAD2 = 0;
    int ref1_y = macro_y + mv1_y;
    int ref1_x = macro_x + mv1_x;
    int ref2_y = macro_y + mv2_y;
    int ref2_x = macro_x + mv2_x;

    for(int i=0; i<BLK_SIZE; i++)
    {
        for(int j=0; j<BLK_SIZE; j++)
        {
            SAD1 += abs((int)ref_frame[ref1_y*width+ref1_x + i*width+j] - (int)curr_frame[macro_y*width+macro_x + i*width+j]);
        }
    }

    for(int i=0; i<BLK_SIZE; i++)
    {
        for(int j=0; j<BLK_SIZE; j++)
        {
            SAD2 += abs((int)ref_frame[ref2_y*width+ref2_x + i*width+j] - (int)curr_frame[macro_y*width+macro_x + i*width+j]);
        }
    }

    if(SAD1 == SAD2)
        return 1;
    
    printf("%d %d ,SAD1: %d, SAD2: %d\n", macro_y, macro_x,SAD1, SAD2);

    return 0;
}

bool compare_mv(uint8_t* ref_frame, uint8_t* curr_frame, int* mv1, int* mv2, size_t len, int width, int BLK_SIZE)
{
    for(int i=0; i<len; i++)
    {
        if(mv1[i] != mv2[i])
        {
            int idx = i/4;
            idx = idx * 4;
            // if(i==16350)
            //     printf("i: %d %d %d %d \n", mv1[idx], mv1[idx+1], mv1[idx+2], mv1[idx+3]);
            if(compare_SAD(ref_frame, curr_frame, mv1[idx], mv1[idx+1], mv1[idx+2], mv1[idx+3], mv2[idx+2], mv2[idx+3], width, BLK_SIZE))
                continue;
            else
                printf("Error in mv: index %d, value1: %d, value2: %d \n", i, mv1[i], mv2[i]);
                return 0;
        }
    }
    return 1;
}


bool read_next_frame(FILE* yuv_file, uint8_t* frame_buffer, size_t WIDTH, size_t HEIGHT, size_t WIDTH_PAD)
{
    // assume frame_buffer is allocated to correct size and filled with 128
    // uint8_t* temp_buffer = (uint8_t *)malloc(frame_bytes);
    size_t result = 0;
    size_t idx = 0;
    size_t frame_bytes = WIDTH*HEIGHT*sizeof(uint8_t);

    if(WIDTH==WIDTH_PAD) // no padding needed
    {
        result = fread(frame_buffer, 1, frame_bytes, yuv_file);
    }
    else // need padding
    {
        for(int i=0; i<HEIGHT; i++)
        {
            idx = i * WIDTH_PAD;
            result += fread((frame_buffer + idx), 1, WIDTH, yuv_file);
        }
    }

    fseek(yuv_file, frame_bytes/2, SEEK_CUR); // skip U, V components
    if (result == frame_bytes) 
        return 1;
    else
        return 0;
}

// usage: main.cu <video file> <WIDTH> <HEIGHT> <BLK_size> <search range>
int main( int argc, char *argv[])
{
    if( argc != 6) {
        printf( "Error: wrong number of args\n");
        exit(1);
    }
    
    char* file_name = argv[1];
    int WIDTH = atoi(argv[2]);
    int HEIGHT = atoi(argv[3]);
    int BLK_SIZE = atoi(argv[4]);
    int SRC_range = atoi(argv[5]);

    // padding handling
    int WIDTH_PAD = (WIDTH + BLK_SIZE - 1)/BLK_SIZE;
    int HEIGHT_PAD = (HEIGHT + BLK_SIZE - 1)/BLK_SIZE;
    WIDTH_PAD = WIDTH_PAD * BLK_SIZE;
    HEIGHT_PAD = HEIGHT_PAD * BLK_SIZE;

    // video file preprocess
    printf("WIDTH_PAD: %d, HEIGHT_PAD: %d \n", WIDTH_PAD, HEIGHT_PAD);

    size_t pixels_pad = WIDTH_PAD*HEIGHT_PAD;
    // size_t pixels = WIDTH*HEIGHT;
    // size_t frame_bytes = pixels*sizeof(uint8_t);
    size_t frame_bytes_pad = pixels_pad*sizeof(uint8_t);
    size_t vector_size = 4*(pixels_pad/BLK_SIZE/BLK_SIZE); // format: y x mv_y mv_x ...
    size_t vector_bytes = vector_size*sizeof(int);

    // process frame
    FILE* yuv_file;
    yuv_file = fopen(file_name, "rb");
    if(yuv_file==NULL) {printf("Error: NULL file \n");exit(0);}
    rewind(yuv_file);

    // Host memory allocation
    uint8_t *h_ref_frame, *h_cur_frame;
    int *h_d_mv; int *h_h_mv;
    gpuErrchk(cudaMallocHost((void **)&h_ref_frame, frame_bytes_pad));
    memset(h_ref_frame, 128, frame_bytes_pad);
    gpuErrchk(cudaMallocHost((void **)&h_cur_frame, frame_bytes_pad));
    memset(h_cur_frame, 128, frame_bytes_pad);
    gpuErrchk(cudaMallocHost((void **)&h_d_mv, vector_bytes));
    gpuErrchk(cudaMallocHost((void **)&h_h_mv, vector_bytes));

    int ret = read_next_frame(yuv_file, h_ref_frame, WIDTH, HEIGHT, WIDTH_PAD);
    // printf("ret = %d \n", ret);
    read_next_frame(yuv_file, h_cur_frame, WIDTH, HEIGHT, WIDTH_PAD);

    // for(int i=0; i < 20; i++)
    //     printf("%hhu ", h_cur_frame[i]);
    // printf("\n");

    dim3 dimGrid((WIDTH_PAD)/BLK_SIZE, (HEIGHT_PAD)/BLK_SIZE);
    dim3 dimBlock(BLK_SIZE, BLK_SIZE);

    // Device memory allocation
    double GPU_start_time=getTimeStamp();

    uint8_t *d_ref_frame, *d_cur_frame;
    int *d_mv;
    cudaMalloc((void **)&d_ref_frame, frame_bytes_pad);
    cudaMalloc((void **)&d_cur_frame, frame_bytes_pad);
    cudaMalloc((void **)&d_mv, vector_bytes);

    cudaMemcpy(d_ref_frame, h_ref_frame, frame_bytes_pad, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cur_frame, h_cur_frame, frame_bytes_pad, cudaMemcpyHostToDevice);

    block_matching<<<dimGrid, dimBlock>>>(d_ref_frame, d_cur_frame, d_mv, WIDTH_PAD, HEIGHT_PAD, BLK_SIZE, SRC_range);

    cudaDeviceSynchronize();

    cudaMemcpy(h_d_mv, d_mv, vector_bytes, cudaMemcpyDeviceToHost);
    double GPU_end_time=getTimeStamp();

    // do compression using h_mv

    // CPU GPU check ////////////
    // double CPU_start_time=getTimeStamp();
    // host_block_matching(h_ref_frame, h_cur_frame, h_h_mv, WIDTH_PAD, HEIGHT_PAD, BLK_SIZE, SRC_range);
    // double CPU_end_time=getTimeStamp();
    // if(compare_mv(h_ref_frame, h_cur_frame, h_h_mv, h_d_mv, vector_size, WIDTH_PAD, BLK_SIZE))
    //     printf("CPU GPU check success\n");
    // else
    //     printf("CPU GPU check failed\n");
        
    float GPU_total_time_ms =(GPU_end_time-GPU_start_time)*1000;
    // float CPU_total_time_ms =(CPU_end_time-CPU_start_time)*1000;
    printf("GPU time: %.4f ms\n", GPU_total_time_ms);
    // printf("CPU time: %.4f ms\n", CPU_total_time_ms);
    /////////////////////////////

    int start = 0;
    for(int i=start; i<start+32; i++)
        printf("%d ", h_d_mv[i]);
    printf("\n");

    // for(int i=start; i<start+32; i++)
    //     printf("%d ", h_h_mv[i]);
    // printf("\n");


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