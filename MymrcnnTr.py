import os
import sys
import cv2
import glob
import math
import random
import skimage.io
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras as kkk

from PIL import Image

import imageio.core.util
def ignore_warnings(*args, **kwargs):
    pass
imageio.core.util._precision_warn = ignore_warnings

print(tf.__version__)
print(kkk.__version__)

ROOT_DIR = os.path.abspath("MaskRCNN/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn.config import Config
import mrcnn.model as modellib
from mrcnn import visualize

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
	utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 8

# ~ config = InferenceConfig()
# ~ config.display()

# ~ # Create model object in inference mode.
# ~ model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
# ~ # Load weights trained on MS-COCO
# ~ model.load_weights(COCO_MODEL_PATH, by_name=True)

class MyConfig(Config):
	"""Configuration for training on the toy shapes dataset.
	Derives from the base Config class and overrides values specific
	to the toy shapes dataset.
	"""
	# Give the configuration a recognizable name
	NAME = "My"
	
	# Train on 1 GPU and 8 images per GPU. We can put multiple images on each
	# GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
	GPU_COUNT = 1
	IMAGES_PER_GPU = 8
	
	# Number of classes (including background)
	NUM_CLASSES = 1 + 1  # background + 2 shapes
	
	# Use small images for faster training. Set the limits of the small side
	# the large side, and that determines the image shape.
	IMAGE_MIN_DIM = 512
	IMAGE_MAX_DIM = 512
	
	# Use smaller anchors because our image and objects are small
	RPN_ANCHOR_SCALES = (16, 32, 64, 128)  # anchor side in pixels
	
	# Reduce training ROIs per image because the images are small and have
	# few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
	TRAIN_ROIS_PER_IMAGE = 8
	
	# Use a small epoch since the data is simple
	STEPS_PER_EPOCH = 8
	
	# use small validation steps since the epoch is small
	VALIDATION_STEPS = 8
    

class MyDataset(utils.Dataset):
	"""Generates the shapes synthetic dataset. The dataset consists of simple
	shapes (triangles, squares, circles) placed randomly on a blank surface.
	The images are generated on the fly. No file access required.
	"""
		
	def add_image_path_and_classes(self, im_path):
					
		the_path = im_path + "/*.bmp"
		print(the_path)
		allfiles =[]
		allfiles = glob.glob(the_path)
		random.shuffle(allfiles)
		# ~ self.image_ids = np.array([len(allfiles)])
		i = 0
		print(len(allfiles),'  ***\n')
		
		for fi in allfiles:
			# ~ self.add_image("Carpets" , i , fi)
			
			# ~ print(i,fi,'\n')
			image_inf = {"id": i, "source": "Carpets", "path": fi , "W": 0 , "H":0, "C":0}
			self.image_info.append(image_inf)
			# ~ print('-------------------  >>>> ', self.image_ids)
			# ~ self.image_ids[i]=i
			
			i += 1
			# ~ if i > 99:
				# ~ break
		print(len(self.image_info),'  ***\n')
			
	def add_class(self, class_names):
		i = 0
		for cl in class_names:
			self.class_info.append({"source": "Carpets", "id": i, "name": cl})
			i += 1
			# ~ print('++++++++++++++++ > >>> ',self.class_info)
		self.num_classes = len(self.class_info)        
	
	def load_image(self, image_id):
			
		info = self.image_info[image_id]
		image = cv2.imread(info['path'])
		h, w, c = image.shape	
		self.image_info[image_id]['W'] = w
		self.image_info[image_id]['H'] = h
		self.image_info[image_id]['C'] = c
		return image
	
		
	def load_mask(self, image_id):
		"""Generate instance masks for shapes of the given image ID. """
		
		info = self.image_info[image_id]
		h = info['H']
		w = info['W']
		c = info['C']
		
		the_path = info['path']
	
		pre, ext = os.path.splitext(info['path'])
		names = pre + "_*.jpg"
		# ~ print('-----------------------> ',names)
		its_masks =[]
		its_masks = glob.glob(names)
		# ~ print('-----------------------> ',its_masks)
		
		mask = np.zeros([h,w,len(its_masks)],dtype=np.uint8)
		class_ids = np.array([len(its_masks)])
		
		i=0
		for ms in its_masks:
			mas = cv2.imread(ms,0)
			th , mas = cv2.threshold(mas , 20 , 255, cv2.THRESH_BINARY)
			mask[:,:,i] = mas
			# ~ plt.imshow(mask[:,:,i],cmap = 'gray')
			# ~ plt.show()
			class_ids[i]=i+1
			i+=1
		# ~ print('Cl1: ',class_ids)
		# ~ print('Cl2: ',class_ids.astype(np.int32))
						
		return mask.astype(np.bool), class_ids.astype(np.int32)
		
	def number_of_data(self):
		return len(self.image_info)
	
        
dataset_train = MyDataset()
# ~ dataset_train.add_class(['Rug', 'Carpet'])
dataset_train.add_class(['Rug'])
dataset_train.add_image_path_and_classes("/home/nrp/Documents/RUGS/4/train")# , {'Rug', 'Carpet'});
dataset_train.prepare()
Ntr = dataset_train.number_of_data()
# ~ dataset_train.load_image(1);
# ~ dataset_train.load_mask(1);

## Validation dataset
dataset_val = MyDataset()
dataset_val.add_class(['Rug'])
dataset_val.add_image_path_and_classes("/home/nrp/Documents/RUGS/4/val")
dataset_val.prepare()
Nval = dataset_val.number_of_data()
# ~ dataset_val.load_image(1);
# ~ dataset_val.load_mask(1);

dataset_test = MyDataset()
dataset_test.add_class(['Rug'])
dataset_test.add_image_path_and_classes("/home/nrp/Documents/RUGS/4/test")
dataset_test.prepare()
Ntest = dataset_test.number_of_data()

config = MyConfig()
config.display()

config.STEPS_PER_EPOCH = Ntr // MyConfig.IMAGES_PER_GPU

print('config.STEPS_PER_EPOCH ::::::::::::::::::::::::::::::::::::::::::::::::: ',config.STEPS_PER_EPOCH)
print('NtrNtrNtrNtrNtrNtrNtrNtr ::::::::::::::::::::::::::::::::::::::::::::::::: ',Ntr)

model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

init_with = "coco" # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)

model.train(dataset_train, 
			dataset_val,
			learning_rate=config.LEARNING_RATE, 
			epochs=1000, 
			layers='heads')                          
			# ~ layers='all')                          

