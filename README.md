# CNNLorenzMie
Localization and estimation of Lorenz-Mie holograms of colloidal spheres using convolutional neural networks


Steps to get setup:
- make darknet 
  In Makefile: GPU=1
               change ARCH to whatever fits your GPU
               nvcc=/path/to/nvcc
               If you're using cuda 9.0 change COMMON and LDFLAGS under GPU=1 to cuda-9.0 paths
- edit holo.data for correct path to names, backup (train/valid don't matter if only using for detection)
