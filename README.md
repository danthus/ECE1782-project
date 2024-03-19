# ECE1782-project

main.cu structure
  read yuv file, 
  read the first Y frame as reference frame
    read the second Y frame as current frame
    pass ref frame and curr frame to GPU kernel
    GPU kernel returns motion vectors
    On the CPU side, use motion vectors to compress current frame
  Update current frame to the next available Y frame until the end of video sequence
  
