# CNNLorenzMie
Localization and estimation of Lorenz-Mie holograms of colloidal spheres using convolutional neural networks


# Setup:
0. Make a fresh environment using conda or venv. (Strongly recommended) Get required packages from `requirements.txt`.

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
		
5. Download darknet [weights file](https://drive.google.com/drive/folders/1Xv8BFnBFSH4pnXnaDXPYy6UtYLjopY1R?usp=sharing). Put it in `CNNLorenzMie/cfg_darknet`.

6. Test if darknet works. In terminal:

	`CNNLorenzMie$ python darknet.py` 
	
   If everything installed properly, you should see a whole bunch of output, with this at the end:
   
   	        106 yolo
		Loading weights from cfg_darknet/holo.weights...Done!

		instantiated
		detecting
		[(b'HOLO', 0.599381685256958, (943.1991577148438, 703.5460205078125, 153.0026092529297, 148.02035522460938)), 		      (b'HOLO', 0.5468334555625916, (331.77569580078125, 478.1768798828125, 94.87818908691406, 94.1760482788086))]

   
7. Install [tensorflow v2.2](https://www.tensorflow.org/install). Keras will be included in the tensorflow installation.

8. Test if keras works. In terminal:

	`CNNLorenzMie$ python3 Estimator.py`
	
   If everything installed properly, you should see a whole bunch of output, with this at the end:
   
   	      Total params: 34,983
	      Trainable params: 34,983
	      Non-trainable params: 0
	      __________________________________________________________________________________________________
	      Example Image:
	      Particle of size 1.065um with refractive index 1.679 at height 157.6
