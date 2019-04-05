# CNNLorenzMie
Localization and estimation of Lorenz-Mie holograms of colloidal spheres using convolutional neural networks


# SETUP:
1. Clone darknet and pylorenzmie repositories (these are the linked submodules but you need to clone them separately)

2. If you don't have GPU, proceed to step 3. Otherwise:

- In darknet Makefile:
  - GPU=1
  - change ARCH to whatever fits your GPU
  - nvcc=/path/to/nvcc
  - If you're using cuda 9.0 change COMMON and LDFLAGS under GPU=1 to cuda-9.0 paths

- Might need to add to your ~/.bashrc:

	       # DARKNET
	       export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
	       export LD_LIBRARY_PATH=/usr/local/cuda9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

3. Make darknet: 

	`CNNLorenzMie/darknet$ make` 
	
   If the make ran correctly, you should see `libdarknet.so` in your darknet directory.
   
4. Edit holo.data for correct path to names file, backup folder (train/valid don't matter if only using for detection)
		
	       classes = 1
	       train = ~/datasets/train/filenames.txt
	       valid = ~/datasets/test/filenames.txt
	       names = /your/path/to/CNNLorenzMie/cfg_darknet/holo.names
	       backup = /your/path/to/CNNLorenzMie/cfg_darknet/
		
5. download darknet [weights file](https://drive.google.com/open?id=1TvffNd64VH0SWM5b7Tkki75SPxo27LUc)
- configure keras, tensorflow

