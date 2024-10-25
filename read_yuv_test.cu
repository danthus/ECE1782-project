#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
using namespace std;

int main()
{
    // video frame parameters
    int WIDTH = 352;
    int HEIGHT = 288;
    size_t frame_bytes = WIDTH*HEIGHT*sizeof(uint8_t);

    // allocation frame buffers
    uint8_t* ref_frame; uint8_t* cur_frame;
    ref_frame = (uint8_t*) malloc(frame_bytes);
    cur_frame = (uint8_t*) malloc(frame_bytes);
    if(ref_frame == NULL) {printf("Error: buffer allocation failed"); exit(0);}

    // open yuv file
    FILE* yuv_file;
    yuv_file = fopen("CIF.yuv", "rb");
    if(yuv_file==NULL) {printf("Error: NULL file \n");exit(0);}
    rewind(yuv_file);

    // read first frame
    size_t result;
    result = fread(ref_frame, 1, frame_bytes, yuv_file);
    if (result != frame_bytes) {printf("Error: file read to buffer failed"); exit (0);}
    fseek(yuv_file, frame_bytes/2, SEEK_CUR); // skip U, V components

    // read next frame
    result = fread(cur_frame, 1, frame_bytes, yuv_file);
    if (result != frame_bytes) {printf("Error: file read to buffer failed"); exit (0);}
    fseek(yuv_file, frame_bytes/2, SEEK_CUR);

    // print elements
    for(int i=0; i<50; i++)
    {
        printf("%hhu ", ref_frame[i]);
    }
    printf("\n");

    FILE* write_file;
    write_file = fopen("y_frames/test_frame", "wb");
    fwrite(ref_frame, 1, frame_bytes, write_file);
    

    free(ref_frame);
    free(cur_frame);
    fclose(yuv_file);
    return 0;
}