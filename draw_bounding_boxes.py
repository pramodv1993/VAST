#!/usr/bin/env python
# coding: utf-8


import numpy as np
import os
import pandas as pd
import ntpath
import cv2 

if __name__ == '__main__':
	if os.path.exists("scripts_assets_datasets\MC2\MC2-Image-Data\MC2-Image-Data_Annotated"):
		for i in range(1,41):
			path = "scripts_assets_datasets\MC2\MC2-Image-Data"

			directory  = 'Person' + str(i)
			path = os.path.join(path, directory)

			for image_path in os.listdir(path):
				input_path = os.path.join(path, image_path)
				if input_path.split('.')[1] == 'jpg' or input_path.split('.')[1] == 'png':
					csv_path = os.path.splitext(input_path)[0] + '.csv'
					pred = pd.read_csv(csv_path)
					font = cv2.FONT_HERSHEY_SIMPLEX 
					fontScale = 5
					color = (0, 255, 255) 
					thickness = 10
					image = cv2.imread(input_path)
					for index, row in pred.iterrows():
						text = row['Label']
						x = row['x']
						y = row['y'] + row['Height']
						org = (x,y)
						cv2.rectangle(image,(row['x'],row['y']),(row['x']+row['Width'],row['y']+row['Height']),(0,0,255),10)
						image = cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA, False) 
					dir_path = 'scripts_assets_datasets\MC2\MC2-Image-Data_Annotated'
					image_name = ntpath.basename(input_path)
					image_fol = image_name.split('_')[0]
					full_path = os.path.join(dir_path, image_fol)
					full_path = os.path.join(full_path, image_name)
					cv2.imwrite(full_path, image)
	else:
		print("Annotated images exists.")



