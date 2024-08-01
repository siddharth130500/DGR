# DGR
Deterministic grid-based resampling as implemented in the paper "Improving LiDAR 3D Object Detection via Range-based Point Cloud Density Optimization" (https://arxiv.org/abs/2306.05663). The LiDAR point cloud from Velodyne is resampled by specifying the new vertical and horizontal angular resolutions. 

The code is implemented in both Python and optimized CUDA versions with multithreading. To run the Python code, run `python dgr.py`. For CUDA version, compile using `nvcc DGR_optimized.cu -o DGR_optimized'. Then run `./DGR_optimized`.

Time comparison between the versions:
* Python version: 62.03323197364807 seconds
* CUDA version: 0.6060945987701416 seconds

Velodyne HDL-64E data is available at https://pdf.directindustry.com/pdf/velodynelidar/hdl-64e-datasheet/182407-676099.html

Deterministic Grid Sampling with Number of Beams = 60, Horizontal resolution = 0.1 degrees:

![alt text](https://github.com/siddharth130500/DGR/blob/main/60_01.png?raw=true)


Deterministic Grid Sampling with Number of Beams = 40, Horizontal resolution = 0.5 degrees:

![alt text](https://github.com/siddharth130500/DGR/blob/main/40_05.png?raw=true)



Original: Number of Beams = 64, Horizontal resolution = 0.08 degrees (Right)
Number of Beams = 16, Horizontal resolution = 0.8 degrees, Range for DGR: 10m (Left)

![alt text](https://github.com/siddharth130500/DGR/blob/main/comp.png?raw=true)


![alt](https://github.com/siddharth130500/DGR/blob/main/dgr.png?raw=true)


