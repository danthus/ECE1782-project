#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

// #define BLK_SIZE 8

double getTimeStamp() {
    struct timeval tv;
    gettimeofday( &tv, NULL );
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}


void host_block_matching(uint8_t* ref_frame, uint8_t* curr_frame, int* mv, int width, int height, int BLK_SIZE, int srch_range)
{
    int minSAD, best_x, best_y;
    int SAD_list[1024];
    int mv_list[2048];
    int val;
    int SAD;

    int macro_y, macro_x, ref_y, ref_x, i, j;

    #pragma omp parallel for private(macro_y, macro_x, ref_y, ref_x, i, j, SAD, minSAD, best_x, best_y) 
    for(macro_y=0; macro_y < height; macro_y+=BLK_SIZE)
    {
        for(macro_x=0; macro_x < width; macro_x+=BLK_SIZE)
        {
            minSAD = 99999;
            for(ref_y=macro_y-srch_range; ref_y < macro_y+srch_range; ref_y++)
            {
                // #pragma omp parallel for private(ref_x)
                for(ref_x=macro_x-srch_range; ref_x < macro_x+srch_range; ref_x++)
                {
                    if(ref_x>=0 && ref_x<=width-BLK_SIZE && ref_y>=0 && ref_y<=height-BLK_SIZE)
                    {
                        SAD = 0;

                        for(i=0; i<BLK_SIZE; i++)
                        {
                            for(j=0; j<BLK_SIZE; j++)
                            {
                                SAD += abs((int)ref_frame[ref_y*width+ref_x + i*width+j] - (int)curr_frame[macro_y*width+macro_x + i*width+j]);
                            }
                            if(SAD >= minSAD)
                                break;
                        }

                        // if(macro_x == 156 && macro_y == 184)
                        //     printf("host: %d %d %d\n", SAD, ref_y-macro_y, ref_x-macro_x);

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
    if( argc != 5) {
        printf( "Error: wrong number of args\n");
        exit(1);
    }
    
    int WIDTH = atoi(argv[1]);
    int HEIGHT = atoi(argv[2]);
    int BLK_SIZE = atoi(argv[3]);
    int SRC_range = atoi(argv[4]);

    // padding handling
    int WIDTH_PAD = (WIDTH + BLK_SIZE - 1)/BLK_SIZE;
    int HEIGHT_PAD = (HEIGHT + BLK_SIZE - 1)/BLK_SIZE;
    WIDTH_PAD = WIDTH_PAD * BLK_SIZE;
    HEIGHT_PAD = HEIGHT_PAD * BLK_SIZE;

    // video file preprocess

    size_t pixels_pad = WIDTH_PAD*HEIGHT_PAD;
    // size_t pixels = WIDTH*HEIGHT;
    // size_t frame_bytes = pixels*sizeof(uint8_t);
    size_t frame_bytes_pad = pixels_pad*sizeof(uint8_t);
    size_t vector_size = 4*(pixels_pad/BLK_SIZE/BLK_SIZE); // format: y x mv_y mv_x ...
    size_t vector_bytes = vector_size*sizeof(int);

    // process frame
    FILE* yuv_file;
    yuv_file = fopen("../CIF.yuv", "rb");
    if(yuv_file==NULL) {printf("Error: NULL file \n");exit(0);}
    rewind(yuv_file);

    // Host memory allocation
    // uint8_t *h_ref_frame, *h_cur_frame;
    // int *h_d_mv; int *h_h_mv;
    // gpuErrchk(cudaMallocHost((void **)&h_ref_frame, frame_bytes_pad));
    // gpuErrchk(cudaMallocHost((void **)&h_cur_frame, frame_bytes_pad));
    // gpuErrchk(cudaMallocHost((void **)&h_d_mv, vector_bytes));
    // gpuErrchk(cudaMallocHost((void **)&h_h_mv, vector_bytes));

    uint8_t* h_ref_frame = (uint8_t*)malloc(frame_bytes_pad);
    uint8_t* h_cur_frame = (uint8_t*)malloc(frame_bytes_pad);
    int* h_h_mv = (int*)malloc(vector_bytes);

    int ret = read_next_frame(yuv_file, h_ref_frame, WIDTH, HEIGHT, WIDTH_PAD);
    // printf("ret = %d \n", ret);
    read_next_frame(yuv_file, h_cur_frame, WIDTH, HEIGHT, WIDTH_PAD);

    // do compression using h_mv

    // CPU GPU check ////////////
    double CPU_start_time=getTimeStamp();
    host_block_matching(h_ref_frame, h_cur_frame, h_h_mv, WIDTH_PAD, HEIGHT_PAD, BLK_SIZE, SRC_range);
    double CPU_end_time=getTimeStamp();

    float CPU_total_time_ms =(CPU_end_time-CPU_start_time)*1000;
    printf("CPU time: %.4f ms\n", CPU_total_time_ms);
    /////////////////////////////

    int start = 0;
    for(int i=start; i<start+32; i++)
        printf("%d ", h_h_mv[i]);
    printf("\n");


    free(h_ref_frame);
    free(h_cur_frame);
    free(h_h_mv);
    fclose(yuv_file);
    return 0;
}