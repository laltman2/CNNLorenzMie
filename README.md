# CNNLorenzMie
Localization and estimation of Lorenz-Mie holograms of colloidal spheres using convolutional neural networks


Steps to get setup:
- In darknet Makefile: GPU=1
               change ARCH to whatever fits your GPU
               nvcc=/path/to/nvcc
	       If you're using cuda 9.0 change COMMON and LDFLAGS under GPU=1 to cuda-9.0 paths

- Might need to add to your ~/.bashrc:

	       # DARKNET
	       export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
	       export LD_LIBRARY_PATH=/usr/local/cuda9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

- make darknet 
- edit holo.data for correct path to names, backup (train/valid don't matter if only using for detection)
- download holo weights file
- configure keras, tensorflow

