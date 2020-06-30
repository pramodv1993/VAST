import dash
import dash_core_components as dcc
import dash_html_components as html
import base64
import plotly.express as px
import numpy as np
import dash_table
import cv2
from skimage import io
from skimage.transform import resize, rotate
import pandas as pd
import os
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from demographic_analysis import Person
from dash.dependencies import Input,Output, State, MATCH
from ClassifierAnalysis import ClassifierAnalysis
import plotly.graph_objects as go

#empty_graph
empty_layout = go.Layout(plot_bgcolor='rgba(0,0,0,0)',
xaxis = dict(showticklabels=False, showgrid=False, zeroline = False),
yaxis = dict(showticklabels = False, showgrid=False, zeroline = False),
height=600, width=800)
empty_graph = go.Figure()
empty_graph.update_layout(empty_layout)

obj = pd.read_csv('Objects.csv')
#initialising all objects True Positives
obj['FLAG'] = ['TP'] * len(obj)
classifier_analysis = ClassifierAnalysis(obj)
DensityVsConfScoreFig = classifier_analysis.GetDensityVsConfScoreGraph()
BBSizeVsConfScoreFig = classifier_analysis.GetBBSizeVsConfScoreGraph()
obj_ground_truth_vs_predictions = classifier_analysis.get_obj_ground_truth_vs_predictions()
uniq_labels = (list(obj.Label.unique()))

label_vs_score = dict()
for label in uniq_labels:
	label_vs_score[label] = obj[obj.Label==label]['conf_score']
group_by_obj = obj.groupby(by=['Label']).mean()['conf_score']
group_by_obj = group_by_obj.reset_index()
#initial bar selection for average conf score bar graph
bar_colors = ['lightslategray'] * len(uniq_labels)
bar_colors[1] = 'crimson'
bar_graph = go.Figure(data=[go.Bar(x=group_by_obj.Label,
y=group_by_obj.conf_score,
marker_color=bar_colors)])
bar_graph.update_layout(title_text='Average Confidence Scores of Objects',
xaxis_title= "Objects",
height=560,
yaxis_title="Confidence Scores")
#Initial bubble selection for the bounding box bubble graph
BBSizeVsConfScoreFig.data[0].marker.color = bar_colors
#initial viz setting
last_selected_obj = bar_graph.data[0].x[1]
last_bubble_idx = 1
last_bar_idx = 1
#initial distribution of scores for an object
df = pd.DataFrame({last_selected_obj: label_vs_score[last_selected_obj]})
dist = ff.create_distplot([df[c] for c in df.columns], df.columns, bin_size=.25)
dist.update_layout(title = "Distribution of Confidence Scores for {}".format(last_selected_obj),
height=560,
xaxis_title = "Scores")
#initalising proportions of objects in Images
objs, counts = classifier_analysis.get_obj_grps_by_count()
objs_proportions_in_imgs_fig = go.Figure(go.Funnelarea(text = objs, values = counts))
objs_proportions_in_imgs_fig.update_layout(showlegend=False, width=600, height=600)
#Totem Analysis 1 - Caption based Connections
person_analysis = Person()
caption_vs_caption = person_analysis.get_caption_to_caption_mapping()
caption_vs_caption_fig = person_analysis.get_caption_2_caption_graph_object(caption_vs_caption,
	title="Person Connectivity based on captions")

#initial distribution of scores of all objects across people
dist2 = person_analysis.getDistribution(last_selected_obj)
#Totem Analysis 2 - Objects Similarity between people
dist_matrix = person_analysis.get_object_similarity_matrix()
heatmap = person_analysis.get_heatmap_fig_for_matrix(dist_matrix)

#external layout
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#init
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#configuring logo
encoded_logo = base64.b64encode(open('tukl.png','rb').read())
encoded_totem = base64.b64encode(open('totem.jpg','rb').read())
#init layout setup

app.layout = html.Div(
	#title and logo
	html.Div
	([
		html.Div(
			[
				html.Div(html.Img(src='data:image/png;base64,{}'.format(encoded_logo.decode()), style={'float':'right'})),
				html.Div(html.Img(src='data:image/png;base64,{}'.format(encoded_totem.decode()), style={'float':'left'}))
			], className='row'
		),
		html.Div(
			[
			html.H1(children='VAST 2020',style={ 'fontSize': 24},className='nine columns')
			], className="row"
		),
		#tabs
		dcc.Tabs([
			#tab1
			dcc.Tab(label='Classifier Analysis', children=
				[
					html.Div([
								#Analysis 1 - bbox vs conf scores
								html.Div([
											dcc.Graph(
											id="BBSizeVsConfScoreFig",
											figure = BBSizeVsConfScoreFig,
										)], className="six columns"),
								#Analysis 2 - density of bboxes vs conf scores
								html.Div([
											dcc.Graph(
											id="DensityVsConfScoreFig",
											figure = DensityVsConfScoreFig,
										)], className="six columns"),
					], className="row"),

					html.Div([
								#Analysis 3 - avg confidence scores of each object
								html.Div([
											dcc.Graph(
											id="bar_graph",
											figure = bar_graph,
										)], className="six columns"),
								#Analysis 4 - distribution of scores of each object across people
								html.Div([
											dcc.Graph(
											id="obj_distribution",
											figure = dist2
										)], className = "six columns")
					], className="row"),

					html.Div([
								#Analysis 5 - Comparison predictions with groud truth - as tablular data
								html.Div([
										html.H1(children='Comparing GroundTruth and Predictions',style={'text-align': 'center', 'fontSize': 20},className='row'),
										html.Div([
											dash_table.DataTable(
												id="pred_table",
												style_data={
												  'whiteSpace': 'normal',
												  'lineHeight': '20px'},
												style_cell={'textAlign': 'left','fontSize':18,  'font-family':'cursive'},
												columns = [{"name":i, "id":i} for i in obj_ground_truth_vs_predictions.columns],
												data = obj_ground_truth_vs_predictions.to_dict('records'),
												editable = True)
											], className="ten columns")

										], className="row"),
								#Analysis 6 - Comparison of test image with train images grid
								html.Div([
									html.H1(children='Comparing Test and Train Images for Objects',style={'margin-top': '25px','text-align': 'center', 'fontSize': 20})]
									, className="row"),
								html.Div([
									html.Div([
										html.H2(children='Test Image',style={'margin-top': '25px','text-align': 'center', 'fontSize': 20}),
										html.H3(id='score_improvement', className = 'row'),
										dcc.Loading(
										id="person_image_loading",
										type='circle',
										children=[
										dcc.Graph(
											id="person_image",
											figure = empty_graph
											)]
										)], className="six columns"),
									html.Div([
										html.H2(children='Train Images',style={'margin-top': '25px','text-align': 'center', 'fontSize': 20}),
										dcc.Dropdown(
											id="grid_drop_down",
											options=[{'label':i, 'value':i} for i in classifier_analysis.obj_ground_truth]
											),
									dcc.Loading(
									id="image_grid_loading",
									type='circle',
									children=[
										dcc.Graph(
										id="image_grid",
										figure = empty_graph
										)]
									)], className = "six columns")
									], className = "row")
					], className="row")
				]),
			#tab2
			dcc.Tab(label='Totem Analysis', children=
				[
					#Analysis 7 - Caption to Caption mapping using a graph and sankey graph
					html.Div([
								#Common words
								html.Div([
									dcc.Graph(
									id="caption_vs_caption",
									figure= caption_vs_caption_fig
									)], className="six columns"),
								html.Div([
									dcc.Graph(
										id="caption_mapping"
									)], className="six columns")
						], className="row"),
					html.Div([
								#Analysis 8 - Objects similarity between people - heatmap
								html.Div([
									dcc.Graph(
									id="obj_similarity",
									figure= heatmap
									)], className="six columns"),
								html.Div([
									dcc.Graph(
									id="similar_objects",
									)], className="six columns"),
							 ], className="row")
				]),
			#Tab3
			dcc.Tab(label='Classifier Correction', children=
				[
					#Analysis 9 - Eliminate FPs through user feedback
					html.Div([
								html.Div([
											html.H1(children='Eliminating False Positives through User Feedback',style={'margin-top': '25px','text-align': 'left', 'fontSize': 25}, className="row"),
											#Eliminate wrongly classified images
											html.H2(children='Objects likely to contain FPs',style={'margin-top': '25px','text-align': 'left', 'fontSize': 20}, className="row"),
											# dcc.Dropdown(
											# options=[{'label': i, 'value': i} for i in classifier_analysis.obj_ground_truth],
											# id='fp_obj_list',
											# style=dict(width='60%')
											# ),
											dcc.Graph(figure = objs_proportions_in_imgs_fig, id='objs_proportions_in_imgs'),
											html.H3(children='Images that are predicted to have the Objects',style={'margin-top': '25px','text-align': 'left', 'fontSize': 20}, className="row"),
											dcc.Dropdown(
											id='fp_img_list',
											style=dict(width='60%'),
											multi = True
											)
										], className="six columns"),
										html.Div([
													html.Div([], id='fp_images', className="grid-container")
												]
										,className="six columns"),
									html.Div([
										html.Button('Eliminate False Positives', id='del_fps', n_clicks=0)
										], className = "row"),
							], className="row"),
						html.Div([
								dcc.Loading(
									id="updated_obj_graph_loading",
									type='circle',
									children = [dcc.Graph(id="updated_obj_graph")]
								)
						],className='six columns'),
					html.Div([], id='dummy')
				])
		])
	])
)

# @app.callback(Output('dummy', 'children'),
# [Input('objs_proportions_in_imgs', 'clickData')])
# def update_fp_images(data):
# 	if data is None:
# 		return []
# 	print(data['points'][0]['text'])


prev_img_list = None
@app.callback(Output('fp_images', 'children'),
	[Input('fp_img_list', 'value')])
def update_fp_images(value):
	"""Usage: Analysis 9; Returns: Updated grid of Images based on object selection"""
	global prev_img_list
	path = "MC2/MC2-Image-Data/{}/{}"
	if value is None or len(value)==0:
		return []
	prev_img_list = value
	images = []
	for person in value:
		img_path = path.format(person.split('_')[0],person)
		encoded_img= base64.b64encode(open(img_path,'rb').read())
		images.append(html.Div([
		html.Img(src='data:image/png;base64,{}'.format(encoded_img.decode()), id={'image_name':person.split('.')[0]}, style={'width':'200px', 'height':'200px'})]
		, className='grid-item'))
	return images


fps_to_eliminate = []
@app.callback(Output('score_improvement', 'children'),
[Input('pred_table','data'),
Input('pred_table','columns')],
[State('pred_table', 'active_cell')])
def compute_improved_score(data, cols, cell):
	"""Usage: Analysis 5; Returns: Computes new average score after removal of FP through User entry in the table"""
	if cell is None:
		return []
	preds = obj_ground_truth_vs_predictions.iloc[cell['row']][cell['column']]
	corrected_preds = data[cell['row']]['Predictions']
	img_name = data[cell['row']]['Image Name']
	img_id = classifier_analysis.images[classifier_analysis.images.name == img_name]['image_id'].iloc[0]
	gt = data[cell['row']]['GroundTruth']
	preds, corrected_preds = preds.split(','), corrected_preds.split(',')
	fp = list(set(preds) - set(corrected_preds))
	if len(fp) > 1:
		print("True value mapping ambiguous!")
	fp = fp[0].strip()
	#unsetting flag for the object for this image
	obj.loc[(obj.Label == fp)&(obj.image_id==img_id),'FLAG'] = 'FP'
	#computing conf score offset
	group_by_obj = obj.groupby(by=['Label']).mean()['conf_score']
	group_by_obj = group_by_obj.reset_index()
	old_conf_score = group_by_obj[group_by_obj.Label==fp]['conf_score'].iloc[0]
	obj_copy = obj.copy()
	obj_copy = obj_copy[obj_copy.FLAG == 'TP']
	group_by_obj_copy = obj_copy.groupby(by=['Label']).mean()['conf_score']
	group_by_obj_copy = group_by_obj_copy.reset_index()
	new_conf_score = group_by_obj_copy[group_by_obj_copy.Label==fp]['conf_score'].iloc[0]
	return ["Average Confidence Score for {} improved by {}".format(fp, float(new_conf_score - old_conf_score))]

highlighted_imgs = list()
@app.callback(Output({'image_name':MATCH},'style'),
[Input({'image_name':MATCH}, 'n_clicks')],
[State({'image_name':MATCH},'id')])
def on_image_click(n_clicks, id):
	"""Usage: Analysis 5; Returns: Computes new average score after removal of FP through User entry in the table"""
	highlight = {'box-shadow':'0px 12px 22px 1px skyblue','width':'200px', 'height':'200px'}
	unhighlight = {'width':'200px', 'height':'200px'}
	global fps_to_eliminate, prev_img_list,highlighted_imgs
	if n_clicks is None:
		return {'width':'200px', 'height':'200px'}
	image_name = id['image_name']
	if image_name in highlighted_imgs:
		print("Removing,", fps_to_eliminate," from ", image_name)
		fps_to_eliminate.remove(image_name)
		prev_img_list.append("{}.jpg".format(image_name))
		highlighted_imgs.remove(image_name)
		return unhighlight
	else:
		prev_img_list.remove("{}.jpg".format(image_name))
		fps_to_eliminate.append(image_name)
		print("Adding", fps_to_eliminate)
		highlighted_imgs.append(image_name)
		return highlight


last_fp_object_selected = None
fp_img_list = []
change_in_conf_scores = dict((label, [0.0]) for label in uniq_labels)
@app.callback([Output('fp_img_list', 'options'),
			   Output('fp_img_list','value'),
			   Output('updated_obj_graph', 'figure')],
			[Input('objs_proportions_in_imgs', 'clickData'),
			Input('del_fps', 'n_clicks')])
def update_fp_images_for_object(value, n_clicks):
	"""Usage: Analysis 9; Returns: Updates images for chosen object while eliminating FPs + adds the FPs to the removal list """
	global last_fp_object_selected, fps_to_eliminate, change_in_conf_scores, prev_img_list, fp_img_list, highlighted_imgs
	init_multi_list_values = []
	if value is None:
		return fp_img_list, prev_img_list, empty_graph
	value = value['points'][0]['text']
	last_fp_object_selected = value
	#unsetting the flags of the false positives
	for person in fps_to_eliminate:
		obj.loc[(obj.Label == last_fp_object_selected)&(obj.name == "{}.csv".format(person)),'FLAG'] = 'FP'
	#preparing the new list of options after eliminating the false positives
	fp_img_list = [{'label':i, 'value':i} for i in classifier_analysis.get_images_for_prediction(value, obj)]
	if n_clicks is None or len(fps_to_eliminate)==0:
		return fp_img_list, init_multi_list_values, empty_graph
	print("Images to delete:", fps_to_eliminate)
	#update the line graph of avg conf score for each object
	group_by_obj = obj.groupby(by=['Label']).mean()['conf_score']
	group_by_obj = group_by_obj.reset_index()
	old_conf_score = group_by_obj[group_by_obj.Label == last_fp_object_selected]['conf_score'].iloc[0]
	obj_copy = obj.copy()
	obj_copy = obj_copy[obj_copy.FLAG == 'TP']
	group_by_obj_copy = obj_copy.groupby(by=['Label']).mean()['conf_score']
	group_by_obj_copy = group_by_obj_copy.reset_index()
	new_conf_score = group_by_obj_copy[group_by_obj_copy.Label == last_fp_object_selected]['conf_score'].iloc[0]
	#tracking new changes in conf scores for objects
	change_in_conf_scores[last_fp_object_selected] += [new_conf_score - old_conf_score]
	conf_scores_for_obj = change_in_conf_scores[last_fp_object_selected]
	line_chart = go.Figure(data=go.Scatter(x = np.arange(1, len(conf_scores_for_obj)+1), y= conf_scores_for_obj),
	layout = go.Layout(yaxis_title= "Average Confidence Score", title = "Change in Avg. Confidence Score for {}".format(last_fp_object_selected),
	 xaxis = dict(showgrid=False, zeroline = False) , yaxis = dict(zeroline = False, showgrid=False)))
	print("Diff in conf score for {} is {}".format(last_fp_object_selected, new_conf_score - old_conf_score))
	#reinitialising fps to eliminate list and highlighted_imgs
	fps_to_eliminate = []
	highlighted_imgs = []
	return fp_img_list, prev_img_list, line_chart


prev_test_img = empty_graph
@app.callback(Output('person_image', 'figure'),
			  [Input('pred_table', 'active_cell')])
def get_active_cell(active_cell):
	"""Usage: Analysis 5; Returns: Updates test image on choosing from the GroundTruth vs Predictions table"""
	global prev_test_img
	path = "MC2/MC2-Image-Data/{}/{}"
	if active_cell is None:
		return prev_test_img
	person = obj_ground_truth_vs_predictions.iloc[active_cell['row'],active_cell['column']]
	#if its an image
	if "." not in person:
		return prev_test_img
	path = path.format(person.split('_')[0],person)
	img = io.imread(path)
	img = resize(img, (200,200))
	prev_test_img = px.imshow(img)
	layout = go.Layout(
	xaxis = dict(showticklabels=False),
	yaxis = dict(showticklabels = False),
	height=800, width=800,
)
	prev_test_img.update_layout(layout)
	return prev_test_img



@app.callback(Output('image_grid', 'figure'),
	[Input('grid_drop_down','value')])
def update_grid(selectedData):
	"""Usage: Analysis 6; Returns: Updates the Training Images grid"""
	if selectedData is None:
		return empty_graph
	fig = px.imshow(io.imread("MC2/grid/{}_grid.png".format(selectedData)))
	layout = go.Layout(
	xaxis = dict(showticklabels=False),
	yaxis = dict(showticklabels = False),
	height=900, width=900,
)
	fig.update_layout(layout)
	return fig

@app.callback(Output('similar_objects', 'figure'),
	[Input('obj_similarity','clickData')])
def update_similar_object_viz(selectedData):
	"""Usage: Analysis 8; Returns: Updates word cloud image, that gives the common objects between people"""
	if selectedData is None:
		return empty_graph
	points = selectedData['points'][0]
	p1, p2 = points['x'], points['y']
	objects = person_analysis.get_common_objects_between(p1,p2)
	if len(objects)==0:
		return empty_graph
	img_array = person_analysis.get_totem_word_cloud(objects)
	fig = px.imshow(img_array)
	layout = go.Layout(
	title = 'Common objects between people',
	xaxis = dict(showticklabels=False),
	yaxis = dict(showticklabels = False),
	height=600, width=800,
)
	fig.update_layout(layout)
	return fig

@app.callback(Output('caption_mapping', 'figure'),
	[Input('caption_vs_caption','selectedData')])
def update_obj_mapping_viz(selectedData):
	"""Usage: Analysis 7; Returns: Updates word cloud image, that gives the common objects between people"""
	if selectedData is None:
		return empty_graph
	points = selectedData['points']
	pids = []
	for point in points:
		person = point['text']
		pids.append(int(person[person.index('#')+1 : person.rindex('|') -1]))
	return person_analysis.get_sankey_diag_for_ppl(caption_vs_caption, pids)

@app.callback([Output('bar_graph','figure'),
	Output('BBSizeVsConfScoreFig', 'figure'),
	Output('obj_distribution', 'figure')],
	[Input('bar_graph','clickData'),
	Input('BBSizeVsConfScoreFig', 'clickData'),
	])
def update_graph_colors_and_dist(bar_data, bubble_data):
	"""Usage: Analysis 1; Returns: Updates bar graph and bubble charts' colors on click"""
	#updating all 3  graphs
	global last_bar_idx, last_bubble_idx, last_selected_obj,bar_graph,BBSizeVsConfScoreFig,dist
	new_idx = 0
	if bar_data is None and bubble_data is None:
		return bar_graph, BBSizeVsConfScoreFig, dist2
	if bar_data is not None:
		curr_bar_idx = bar_data['points'][0]['pointIndex']
		if curr_bar_idx != last_bar_idx:
			print("bar",last_bar_idx, curr_bar_idx)
			new_idx = curr_bar_idx
			last_bar_idx = new_idx
			last_selected_obj  = bar_data['points'][0]['x']
	if bubble_data is not None:
		curr_bubble_idx = bubble_data['points'][0]['pointIndex']
		if curr_bubble_idx != last_bubble_idx:
			print("bubble",last_bubble_idx, curr_bubble_idx)
			new_idx = curr_bubble_idx
			last_bubble_idx = new_idx
			last_selected_obj  = bubble_data['points'][0]['customdata'][0]
	updated_colors = ['lightslategray'] * len(uniq_labels)
	updated_colors[new_idx] = 'crimson'
	#update bar graph
	bar_graph.data[0].marker.color = updated_colors
	#update bubble chart
	BBSizeVsConfScoreFig.data[0].marker.color = updated_colors
	#update distribution
	# df = pd.DataFrame({last_selected_obj: label_vs_score[last_selected_obj]})
	# dist = ff.create_distplot([df[c] for c in df.columns], df.columns, bin_size=.25)
	# dist.update_layout(title = "Distribution of Confidence Scores for {}".format(last_selected_obj),
	# height=560,
	# xaxis_title = "Scores")
	#update distribution across people
	return bar_graph, BBSizeVsConfScoreFig, person_analysis.getDistribution(last_selected_obj)

if __name__ == '__main__':
	app.run_server(debug=True)
