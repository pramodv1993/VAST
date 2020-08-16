import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
train_path = 'scripts_assets_datasets/MC2/MC2-Image-Data/TrainingImages/'
output_path = 'scripts_assets_datasets/MC2/grid/'
objects = os.listdir(train_path)
if __name__=="__main__":
	if not os.path.exists(output_path):
		for obj in objects:
			print("Creating a grid for obj: {}".format(obj))
			img_files = os.listdir(os.path.join(train_path, obj))
			images = [cv2.imread(os.path.join(train_path,obj, img_file)) for img_file in img_files if img_file != 'Thumbs.db']
			#create grid
			fig = plt.figure()
			grid = ImageGrid(fig, 111,
						 nrows_ncols=(3, 4),  
						 axes_pad=0,
						 )
			for ax, im in zip(grid, images):
				ax.imshow(im)
				ax.get_xaxis().set_visible(False)
				ax.get_yaxis().set_visible(False)
			plt.savefig(os.path.join(output_path,'_'.join([obj, 'grid'])), transparent = True)
	else:
		print("Image grids exists.")
	