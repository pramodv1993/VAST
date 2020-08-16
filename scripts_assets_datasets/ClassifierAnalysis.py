import pandas as pd
import numpy as np
import plotly.express as px
import os
from text_processing_service import TextProcessing
# Class for handling the classifier analysis
class ClassifierAnalysis():
	def __init__(self, objects):
			self.objectDB = objects
			self.images = pd.read_csv('Images.csv')
			self.obj = pd.read_csv('Objects.csv')
			self.obj_ground_truth = dict((obj.lower(), (obj, i)) for i,obj in enumerate(sorted(os.listdir('./MC2/MC2-Image-Data/TrainingImages/'))))
			txt_process = TextProcessing()
			processed_captions = []
			for caption in self.images.caption:
				if caption == "NO_CAPTION":
					processed_captions.append(["NO_CAPTION"])
				else:
					processed_captions.append(txt_process.process_sentence(caption))
			self.images['processed_caption'] = processed_captions

	def CalculateBoundingBoxSize(self, row):
		size = row['width'] * row['height']
		return size

	def FindCentroid(self, row):
		x = row['x'] + (row['width'] / 2)
		y = row['y'] + (row['height'] / 2)
		return pd.Series([x, y])

	def GetBBSizeVsConfScoreGraph(self):
		self.objectDB['BoundingBoxSize'] = self.objectDB.apply(self.CalculateBoundingBoxSize, axis = 1)
		group = self.objectDB.groupby('Label')

		avg_bb_size = list()
		conf_score_mean = list()
		conf_score_var = list()
		object_name = list()
		for i, grp in group:
			bb_Size = grp['BoundingBoxSize'].mean()
			conf_mean = grp['conf_score'].mean()
			var_conf = grp['conf_score'].var()
			avg_bb_size.append(bb_Size)
			conf_score_mean.append(conf_mean)
			conf_score_var.append(var_conf)
			object_name += [i]

		df = pd.DataFrame(object_name, columns=['object_name'])
		df['conf_score_mean'] = conf_score_mean
		df['conf_score_var'] = conf_score_var
		df['avg_bb_size'] = avg_bb_size

		fig = px.scatter(df, x="conf_score_mean", y="conf_score_var", size="avg_bb_size", hover_data=["object_name"])
		fig.update_layout(title='Mean vs Variance of Confidence scores for each object',
		plot_bgcolor="#F9F9F9",
		paper_bgcolor="#F9F9F9",
		xaxis_title="Conf_Score mean",
		yaxis_title="Conf_Score Variance")
		return fig

	def GetDensityVsConfScoreGraph(self):
		group = self.objectDB.groupby('name')

		average_density = list()
		average_conf_score = list()
		image_name = list()
		for name, grp in group:
			xmin = grp['x'].min()
			xmax = grp['x'].max()
			ymin = grp['y'].min()
			ymax = grp['y'].max()
			size = (xmax - xmin) * (ymax - ymin)
			with np.errstate(divide='ignore'):
				avg_density = len(grp) / size
				average_density.append(avg_density)
			average_conf_score.append(grp['conf_score'].mean())
			image_name += [name]

		df = pd.DataFrame(image_name, columns=['name'])
		df['average_density'] = average_density
		df['average_conf_score'] = average_conf_score

		df = df.replace([np.inf, -np.inf], np.nan)
		df = df.dropna()

		fig = px.scatter(df, "average_density", y="average_conf_score",  hover_data=["name"])
		fig.update_layout(title = 'Mean Confidence Score vs Average density of each object in the images ',
		plot_bgcolor="#F9F9F9",
		paper_bgcolor="#F9F9F9",
		xaxis_title="Average Density of Objects",
		yaxis_title="Conf_Score Mean")
		fig.update_xaxes(range=[0.00001, 0.0003])
		return fig

	def get_obj_grps_by_count(self):
		obj_grp_by_cnt = self.obj.groupby(by='Label').count().iloc[:,:1]
		obj_grp_by_cnt = obj_grp_by_cnt.to_dict()[obj_grp_by_cnt.columns[0]]
		obj_grp_by_cnt = sorted(obj_grp_by_cnt.items(), key=lambda x: x[1], reverse= True)
		objs, counts = [tup[0] for tup in obj_grp_by_cnt], [tup[1] for tup in obj_grp_by_cnt]
		return objs, counts

	def get_obj_ground_truth_vs_predictions(self):
		obj_vs_predictions = list()
		for row in self.images.iterrows():
			row = row[1]
			processed_caption = row['processed_caption']
			image_id = row['image_id']
			image_name = row['name']
			if "NO_CAPTION" in processed_caption:
				continue
			for token in processed_caption:
				if token.lower() in self.obj_ground_truth.keys():
					predictions = list(set(self.obj[self.obj.image_id==image_id]['Label']))
					obj_vs_predictions.append([token.lower(), image_name, ' , '.join(predictions)] )
		return pd.DataFrame(obj_vs_predictions, columns = ['GroundTruth','Image Name', 'Predictions']).drop_duplicates()

	def get_images_for_prediction(self, pred, updated_objs):
		image_ids = updated_objs[(updated_objs.Label== pred) & (updated_objs.FLAG=='TP')]['image_id']
		return self.images[self.images.image_id.isin(image_ids)]['name']
