int w, h; // Image width and height
int blockWidth, blockHeight; // Block width and height
int numBlocksX = w / blockWidth; // Number of blocks in x direction
int numBlocksY = h / blockHeight; // Number of blocks in y direction
int totalBlocks = numBlocksX * numBlocksY; // Total number of blocks

float step = 0.5; // Search grid step size
int searchWidth, searchHeight; // Search window dimensions
int gridPointsX = searchWidth / step; // Number of search grid points in x direction
int gridPointsY = searchHeight / step; // Number of search grid points in y direction
int totalGridPoints = gridPointsX * gridPointsY; // Total number of search grid points

int* value; // Array to store the result values
texture<float, 2> ref_img; // Reference image texture
texture<float, 2> cur_img; // Current image texture


// Kernel function for exhaustive search
__global__ void exhaustiveSearchKernel(int *value, int w, int h, int blockWidth, int blockHeight, int totalBlocks, int searchWidth, int searchHeight, int gridPointsX, int gridPointsY, int totalGridPoints) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index
    int blockID = idx / totalGridPoints; // ID of the block being processed
    int gridIdx = idx % totalGridPoints; // Index within the search grid

    // Return if the block ID is beyond the total number of blocks
    if (blockID >= totalBlocks) return;

    // Calculate the top-left corner of the block in the image
    float blockX = (blockID / numBlocksY) * blockWidth + 0.5f;
    float blockY = (blockID % numBlocksY) * blockHeight + 0.5f;

    // Compute the displacement based on the search grid index
    float offsetX = ((gridIdx / gridPointsX) - gridPointsX / 2) * searchWidth / gridPointsX * step;
    float offsetY = ((gridIdx % gridPointsY) - gridPointsY / 2) * searchHeight / gridPointsY * step;

    float diffSum = 0; // Sum of absolute differences
    // Iterate over each pixel in the block
    for (int i = 0; i < blockWidth; ++i) {
        for (int j = 0; j < blockHeight; ++j) {
            // Accumulate the absolute difference between the reference and current image pixels
            diffSum += abs(tex2D(ref_img, blockX + i, blockY + j) - tex2D(cur_img, blockX + i + offsetX, blockY + j + offsetY));
        }
    }

    // Store the accumulated difference in the result array
    value[idx] = (int)diffSum;
}
