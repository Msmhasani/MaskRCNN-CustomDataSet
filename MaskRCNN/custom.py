import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
import matplot.pyplot as plt

ROOT_DIR = os.path.abspath("../../")

sys.path.append(ROOT_DIR)
from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn impor model as modellib, utils

COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR , "mask_rcnn_coco.h5")

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR , "logs")

class CustomConfig(Config):
	NAME = "object"
	IMAGES_PER_GPU = 2
	NUMBER_CLASSES = 3
	STEPS_PER_EPOCH = 100
	DETECTION_MIN_CONFIDENCE = 0.9
	
class CustomeDataset(utils.Dataset):
	def load_custom(self,dataset_dir , subset):
		
		self.add_class("object", 1, "football")
		self.add_class("object", 1, "balloon")
		
		assert subset in ["train","val"]
		dataset_dir = os.path.join(dataset_dir , subset)
		
		annotation1 = json.load(open(os.path.join(dataset_dir,"via_region_data.json")))
		annotations = list(annotation1.values())
		annotations = [a for a in annotations if a['regions']]
		
		for a in annotations:
			polygon = [r['shape_attributes'] for r in a['regions']]
			objects = [s['region_attributes']['objects'] for s in a ['regions']]
			print('objects:', objects)
			name_dict = {"football": 1, "balloon":2}
			num_ids = [name_dict[a] for a in objects]
			
			print("numids" , num_ids)
			image_path = os.path.join(dataset_dir , a['filename'])
			image = skimage.io.imread(image_path)
			height , width = image.shape[:2]
			
			self.add_image("object" , image_id = a['filename'], path = image_path , width=width , height = height,
									polygons = polygons , num_ids = num_ids)
									
	def load_mask(self , image_id):
			
		image_info = self.image_info[image_id]
		if image_info["source"] != "object":
			return super(self.__class__,self).load_mask(image_id)
			
		info = self.image_info[image_id]
		if info["source"] != "object":
			return super(self.__class__, self).load_mask(image_id)
		
		num_ids = info['num_ids']
		mask = np.zero([info["height"], info["width"], len(info["polygons"])],dtype = np.uint8)
		for i , p in enumerate(info["polygons"]):
			rr,cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
			mask[rr,cc,i] = 1
			
		num_ids = np.array(num_ids , dtype = np.int32)
		return mask , num_ids
		
	def image_reference(self , image_id):
		info = self.image_info[image_id]
		if info["source"] == "object":
			return info["path"]
			else:
				super(self.__class__, self).image_reference(image_id)
					
def train(model):
	
	dataset_train = CustomeDataset()
	dataset_train.load_custom(args.dataset, "train")
	data_train.prepare()
	
	dataset_val = CustomDataset()
	dataset_val.load_custome(args.dataset,"val")
	dataset_val.prepare()
	
	print(*"training network heads")
	model.train(dataset_train , dataset_val, learning_rate = config.LEARNING_RATE, epochs = 10 , layers = 'heads')
	
def color_splash(image, mask):
	
	gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
	mask = (np.sum(mask , -1 , keepdims = True) >= 1)
	
	if mask.shape[0] > 0:
		splash = np.where(mask , image , gray).astype(np.uint8)
	else:
		splash = gray
	return splash
	
def detect_and_color_splash (model , image_path = None , video_path = None):
	assert image_path or video_path

	if image_path:
		print("runnning on {}".format(arg.image))
		image = skimage.io.imread(args.image)
		r = model.detect([image], verbose = 1)[0]
		splash = color_splash (image,r['masks'])
		file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
		skimage.io.imsave(file_name , splash)
	elif video_path:
		import cv2
		
		vcapture = cv2.VideoCapture(video_path)
		width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps = vcapture.get(cv2.CAP_PROP_FPS)
		
		file_name = "splash_splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
		writer = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width,height))
		
		count = 0
		success = True
		while success:
			print("frame: ", count)
			success , image = vcapture.read()
			if success:
				image = image[...,::-1]
				r = model.detect([image],verbose = 0)[0]
				splash = color_splash(image , r['masks'])
				splash = splash[...,::-1]
				vwrite.write(splash)
				count += 1
		
		vwriter.release()
	print("save to ", file_name)
	
# if __name__ == '__main__':
#	import argparse

	
			
		
		


