import numpy as np
import pandas as pd
from ast import literal_eval
import plotly.graph_objects as go
import networkx as nx
from text_processing_service import TextProcessing
from sklearn.metrics.pairwise import euclidean_distances
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import plotly.express as px

from wordcloud import WordCloud, STOPWORDS

class Person:

	def __init__(self):
		self.images = pd.read_csv('Images.csv')
		self.text = pd.read_csv('Texts.csv')
		self.obj = pd.read_csv('Objects.csv')
		self.dist = pd.read_csv('ObjectDistribution.csv')
		self.dist['Distribution'] = self.dist.apply(self.ConvetToArray, axis = 1)
		self.uniq_labels = [label.lower() for label in self.obj.Label.unique()]
		txt_process = TextProcessing()
		processed_captions = []
		for caption in self.images.caption:
			if caption == "NO_CAPTION":
				processed_captions.append(["NO_CAPTION"])
			else:
				processed_captions.append(txt_process.process_sentence(caption))
		self.images['processed_caption'] = processed_captions
		#empty_graph
		empty_layout = go.Layout(
		plot_bgcolor="#F9F9F9",
		paper_bgcolor="#F9F9F9",
		title = 'Common objects between persons',
		xaxis = dict(showticklabels=False, showgrid=False, zeroline = False),
		yaxis = dict(showticklabels = False, showgrid=False, zeroline = False),
		height=600, width=800,
		)
		self.empty_graph = go.Figure()
		self.empty_graph.update_layout(empty_layout)

	def get_caption_to_caption_mapping(self):
		personid_vs_caption = dict()
		uniq_pids = self.obj.person_id.unique()
		for pid in uniq_pids:
			 personid_vs_caption[pid] = personid_vs_caption.get(pid, [])+ list(self.images[self.images.person_id ==pid]['processed_caption'])
		caption_vs_caption = dict()
		for row in self.images.iterrows():
			row = row[1]
			from_caption = row['processed_caption']
			from_id = row['person_id']
			if "NO_CAPTION" in from_caption:
				continue
			for to_id, to_captions in personid_vs_caption.items():
				for to_caption in to_captions:
					if "NO_CAPTION" in to_caption:
						continue
					elif to_id==from_id:
						continue
					new_common_words = list(set(from_caption).intersection(set(to_caption)))
					if len(new_common_words)>0:
						if from_id in caption_vs_caption:
							common_group = caption_vs_caption[from_id]
							if to_id in common_group.keys():
								common_words = common_group[to_id]
								common_words.extend(new_common_words)
								common_group[to_id] = list(set(common_words))
							else:
								common_group[to_id] = new_common_words
							caption_vs_caption[from_id] = common_group
						else:
							caption_vs_caption[from_id] = dict({pid: new_common_words})
		return caption_vs_caption

	def get_object_similarity_matrix(self):
		pids = self.obj.person_id.unique()
		uniq_labels = [label.lower() for label in self.obj.Label.unique()]
		obj_mat = []
		for pid in pids:
			vector = pd.DataFrame(np.zeros(len(uniq_labels)), index = uniq_labels)
			labels =list( self.obj[self.obj.person_id == pid]['Label'])
			for label in uniq_labels:
				vector.loc[label] = labels.count(label)
			obj_mat.append(np.ravel(vector))
		return euclidean_distances(obj_mat)

	def get_heatmap_fig_for_matrix(self, matrix):
		fig =go.Figure(data = go.Heatmap( z= matrix))
		fig.update_layout(height = 700,
		plot_bgcolor="#F9F9F9",
		paper_bgcolor="#F9F9F9",
		width=800,
		title="Object Similarity between Persons")
		return fig


	def get_sankey_diag_for_ppl(self, caption_vs_caption ,pids):
		persons = []
		words = []
		if len(pids) ==1:
			if pids[0] in caption_vs_caption.keys():
				import itertools
				words  += set(list(itertools.chain(*caption_vs_caption[pids[0]].values())))
				persons += [pids[0]] * len(words)
			else:
				return self.empty_graph.update_layout(title="Common words between persons")
		else:
			for _from in pids:
				if _from not in caption_vs_caption.keys():
						continue
				for _to in pids:
					if _from ==_to or _to not in caption_vs_caption[_from].keys():
						continue
					for obj in self.uniq_labels:
						for word in caption_vs_caption[_from][_to]:
							if word.lower() in obj:
								words += [word]
								persons += [_from]
		print("Common objects {}".format(words))
		fig = go.Figure(go.Parcats(
			dimensions=[
				{'label': 'Persons',
				 'values': persons},
				{'label': 'Common Words',
				 'values': words}],
				 line={'colorscale': [[0, 'gray'], [1, 'firebrick']], 'cmin': 0,
					  'cmax': 1, 'color': np.zeros(len(persons)), 'shape': 'hspline'}))
		fig.update_layout(height = 800,
		plot_bgcolor="#F9F9F9",
		paper_bgcolor="#F9F9F9",
		title="Common words between persons")
		return fig

	def get_common_objects_between(self, p1, p2):
		return set(self.obj[self.obj.person_id==p1].Label).intersection(set(self.obj[self.obj.person_id==p2].Label))

	def get_caption_2_caption_graph_object(self, caption_vs_caption, title ="Network Connections"):
		edge_x = []
		edge_y = []
		G = nx.Graph()
		G.add_nodes_from(np.arange(1,41))
		edges = [G.add_edge(from_id,to_id,weight = len(common_words))
		for from_id in sorted(caption_vs_caption) for to_id, common_words in caption_vs_caption[from_id].items()]
		#adding positions
		pos = nx.spring_layout(G, k=2.5)
		for n, p in pos.items():
			G.nodes[n]['pos'] = p
		for edge in G.edges():
			x0, y0 = G.nodes[edge[0]]['pos']
			x1, y1 = G.nodes[edge[1]]['pos']
			edge_x.append(x0)
			edge_x.append(x1)
			edge_x.append(None)
			edge_y.append(y0)
			edge_y.append(y1)
			edge_y.append(None)

		edge_trace = go.Scatter(
			x=edge_x, y=edge_y,
			line=dict(width=0.5, color='#888'),
			hoverinfo='none',
			mode='lines')

		node_x = []
		node_y = []
		for node in G.nodes():
			x, y = G.nodes[node]['pos']
			node_x.append(x)
			node_y.append(y)


		node_trace = go.Scatter(
			x=node_x, y=node_y,
			mode='markers',
			hoverinfo='text',
			marker=dict(
				showscale=True,
				colorscale='YlGnBu',
				reversescale=True,
				color=[],
				size=10,
				colorbar=dict(
					thickness=15,
					title='Node Connections',
					xanchor='left',
					titleside='right'
				),
				line_width=2))

		node_adjacencies = []
		node_text = []
		for node, adjacencies in enumerate(G.adjacency()):
			node_adjacencies.append(len(adjacencies[1]))
			node_text.append('Person #{} | # of connections: {}'.format(node+1,str(len(adjacencies[1]))))

		node_trace.marker.color = node_adjacencies
		node_trace.text = node_text

		fig = go.Figure(data=[edge_trace, node_trace],
					 layout=go.Layout(
						title=title,
						titlefont_size=16,
						showlegend=False,
						hovermode='closest',
						margin=dict(b=20,l=5,r=5,t=40),
						xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
						yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
						)
		fig.update_layout(height = 700)
		return fig


	def get_totem_word_cloud(self, words):
		# generate word cloud
		if len(words) ==0:
			return self.empty_graph
		wc = WordCloud(background_color='#F9F9F9').generate(' '.join(words))
		return wc.to_array()

	def ConvetToArray(self, row):
    		return np.array(literal_eval(row['Distribution']))

	def getDistribution(self, ObjectName):
    		row = self.dist.loc[self.dist['Object']== ObjectName]
    		fig = px.bar(x=np.arange(1,41), y=row.iloc[0]['Distribution'])
    		fig.update_layout(title="Distribution of {} across 40 people".format(ObjectName),
			plot_bgcolor="#F9F9F9",
			paper_bgcolor="#F9F9F9",
			xaxis_title="Pesron Id",
			yaxis_title="Count")
    		return fig
