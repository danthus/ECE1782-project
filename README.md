# `main.cu` structure for ECE1782-project

The `main.cu` file is structured as follows:

- **Read YUV file**: The program starts by reading a YUV file.
- **Read the first Y frame as reference frame**: It then reads the first Y frame from the file to use as a reference frame.
  - **Read the second Y frame as current frame**: Subsequently, it reads the second Y frame to serve as the current frame.
  - **Process frames in GPU**:
    - The reference frame and current frame are passed to a GPU kernel.
    - The GPU kernel processes these frames and returns motion vectors.
  - **Compress current frame**:
    - On the CPU side, these motion vectors are used to compress the current frame, implementing the compression logic.
- **Iterate through Y frames**:
  - The current frame is updated to the next available Y frame, and this process continues until the end of the video sequence.
