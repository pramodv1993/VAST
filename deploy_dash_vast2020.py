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
empty_layout = go.Layout(plot_bgcolor="#F9F9F9",
paper_bgcolor="#F9F9F9",
xaxis = dict(showticklabels=False, showgrid=False, zeroline = False),
yaxis = dict(showticklabels = False, showgrid=False, zeroline = False),
height=700, width=700)
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

#stores lower left and upper right coordinates
BoundingBox = np.array([0,0,0,0])
lastImageSelected = 'Person1_1'

#init image for annotation
image = io.imread('MC2\MC2-Image-Data\Person1\Person1_1.jpg')
image = resize(image, (image.shape[0] // 20, image.shape[1] // 15), anti_aliasing=True)
ImageFigureInit = px.imshow(image)
ImageFigureInit.update_layout(dragmode='drawrect',newshape=dict(line_color='cyan'))

x_vals = list()
y_vals = list()
width = list()
height = list()
newObject = list()

#initial bar selection for average conf score bar graph
bar_colors = ['lightslategray'] * len(uniq_labels)
bar_colors[1] = 'crimson'
bar_graph = go.Figure(data=[go.Bar(x=group_by_obj.Label,
y=group_by_obj.conf_score,
marker_color=bar_colors)])
bar_graph.update_layout(title_text='Average Confidence Scores of Objects',
plot_bgcolor="#F9F9F9",
paper_bgcolor="#F9F9F9",
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
plot_bgcolor="#F9F9F9",
paper_bgcolor="#F9F9F9",
xaxis_title = "Scores")
#initalising proportions of objects in Images
objs, counts = classifier_analysis.get_obj_grps_by_count()
objs_proportions_in_imgs_fig = go.Figure(go.Funnelarea(text = objs, values = counts))
objs_proportions_in_imgs_fig.update_layout(showlegend=False,
plot_bgcolor="#F9F9F9",
paper_bgcolor="#F9F9F9",
width=600, height=600)
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
encoded_dept_logo = base64.b64encode(open('dept_logo.png','rb').read())
#init layout setup

app.layout = html.Div(
	#title and logo
	html.Div
	([
		html.Div(
			[
				html.Div(html.Img(src='data:image/png;base64,{}'.format(encoded_dept_logo.decode()), style={'width':'378px','height':'158px', 'float':'left'})),
				html.Div(html.Img(src='data:image/png;base64,{}'.format(encoded_logo.decode()), style={'width':'328px','height':'158px','float':'right'}))
			], className='row'
		),
		html.Div(
			[
			html.Br(),
			html.H1(children='VAST Challenge 2020',style={ 'fontSize': 34, 'text-align':'center'})
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
										)], className="pretty_container six columns"),

								html.Div([
												#Analysis - distribution of confidence Scores
												dcc.Graph(
														id="score_distribution",
														figure = dist
												)], className = "pretty_container six columns")

					], className="row"),

					html.Div([
								#Analysis 3 - avg confidence scores of each object
								html.Div([
											dcc.Graph(
											id="bar_graph",
											figure = bar_graph,
										)], className="pretty_container six columns"),
								#Analysis 4 - distribution of scores of each object across people
								html.Div([
											dcc.Graph(
											id="obj_distribution",
											figure = dist2
										)], className = "pretty_container six columns")
					], className="row"),
					html.Div([#Analysis 2 - density of bboxes vs conf scores
								dcc.Graph(
								id="DensityVsConfScoreFig",
								figure = DensityVsConfScoreFig,
							)], className="pretty_container six columns"),
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
									)], className="pretty_container six columns"),
								html.Div([
									dcc.Graph(
										id="caption_mapping"
									)], className="pretty_container six columns")
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
							 ], className="pretty_container row")
				]),
			#Tab3
			dcc.Tab(label='Classifier Correction I', children=
				[
					#Analysis 9 - Eliminate FPs through user feedback
					html.Div([
								html.Div([
											html.H1(children='Eliminating False Positives through User Feedback',style={'margin-top': '25px','text-align': 'left', 'fontSize': 25}, className="row"),
											#Eliminate wrongly classified images
											html.H2(children='Select object likely to contain FPs',style={'margin-top': '25px','text-align': 'left', 'fontSize': 20}, className="row"),
											dcc.Graph(figure = objs_proportions_in_imgs_fig, id='objs_proportions_in_imgs'),
											html.H3(children='Images that are predicted to have the selected Object',style={'margin-top': '25px','text-align': 'left', 'fontSize': 20}, className="row"),
											dcc.Dropdown(
											id='fp_img_list',
											style=dict(width='60%'),
											multi = True
											)
										], className="pretty_container six columns"),
										html.Div([
													html.H3("Images"),
													html.Div([], id='fp_images', className="grid-container")
												]
										,className="pretty_container six columns"),
							], className="row"), html.Br(),
					html.Div([
									html.Div(html.Button('Eliminate False Positives', id='del_fps', n_clicks=0)
									,className='pretty_container four columns'),
									# html.Div(html.Button('Re-annotate Images', id='reannotate_tps', n_clicks=0)
									# ,className='six columns')
							]
						,className="row"),
					html.Div([
								dcc.Loading(
									id="updated_obj_graph_loading",
									type='circle',
									children = [dcc.Graph(id="updated_obj_graph")]
								)
						],className='pretty_container row'),
					#Analysis 10 - Reannotation of TPs
					html.Div([html.Div([
								html.H3(children='Re-annotate the ambiguous images',style={'margin-top': '25px','text-align': 'center', 'fontSize': 20}),
									dcc.Dropdown(
    								options=[
        									{'label': 'Person1_1', 'value': 'Person1_1'},
            								{'label': 'Person1_2', 'value': 'Person1_2'},
            								{'label': 'Person1_3', 'value': 'Person1_3'}
    									],
    									id='TruePositiveImages',
									)],className = "pretty_container row"),
								html.Div([
									html.Div([
										dcc.Graph(
										id="ImageFigure",
										config={'modeBarButtonsToAdd':['drawline',
	                                        'drawopenpath',
	                                        'drawclosedpath',
	                                        'drawcircle',
	                                        'drawrect',
	                                        'eraseshape'
	                                       ]}
										)], className="pretty_container six columns"),
									html.Div([], id='BBimage', className='pretty_container six columns')
									], className='row')
								]),
					html.Div([
								html.Div([
									html.Button('Reannotate', id='done-val', n_clicks=0)], className = "six columns"),
								html.Div([
									html.Div(id='dummy')], className = "four columns"),
								html.Div([
									html.Div("Select new objects.."),
									dcc.Dropdown(
    								options=[
        									{'label': i, 'value': i} for i in classifier_analysis.obj_ground_truth
    									],
    									id='CorrectedObjects',
										multi = True
									)],className = "four columns")
							], className="pretty_container row"),
					dcc.ConfirmDialog(id='Exported',message='File exported and ready to be sent for Training..')
				]),
			#Tab4
			dcc.Tab(label='Classifier Correction II', children=
			[
				#Analysis 5 - Correction via comparison with the ground truth
				html.Div([
							html.Div([
									html.H1(children='Comparing GroundTruth and Predictions',style={'padding-top':'20px', 'text-align': 'center', 'fontSize': 20},className='row'),
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
										], className="pretty_container ten columns")

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
									html.Div([], id="person_image")]
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
									html.Div([], id='image_grid')
									]
								)], className = " six columns")
								], className = "pretty_container row")
				], className="row")
			])
		])
	])
)

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
		fps_to_eliminate.remove(image_name)
		prev_img_list.append("{}.jpg".format(image_name))
		highlighted_imgs.remove(image_name)
		return unhighlight
	else:
		prev_img_list.remove("{}.jpg".format(image_name))
		fps_to_eliminate.append(image_name)
		highlighted_imgs.append(image_name)
		return highlight

prev_tps_list = []
# @app.callback(Output('TruePositiveImages', 'options'),
# [Input('reannotate_tps', 'n_clicks')])
# def reannotate(n_clicks):
# 	global prev_tps_list
# 	if n_clicks is None:
# 		return prev_tps_list
# 	prev_tps_list = [{'label': val, 'value': val} for val in fps_to_eliminate]
# 	return prev_tps_list

last_fp_object_selected = None
fp_img_list = []
change_in_conf_scores = dict((label, [0.0]) for label in uniq_labels)
@app.callback([Output('fp_img_list', 'options'),
			   Output('fp_img_list','value'),
			   Output('updated_obj_graph', 'figure'),
			   Output('TruePositiveImages', 'options')],
			[Input('objs_proportions_in_imgs', 'clickData'),
			Input('del_fps', 'n_clicks')])
def update_fp_images_for_object(value, n_clicks):
	"""Usage: Analysis 9; Returns: Updates images for chosen object while eliminating FPs + adds the FPs to the removal list """
	global last_fp_object_selected, fps_to_eliminate, change_in_conf_scores, prev_img_list, fp_img_list, highlighted_imgs
	tps_list = []
	init_multi_list_values = []
	if value is None:
		return fp_img_list, prev_img_list, empty_graph, tps_list
	value = value['points'][0]['text']
	last_fp_object_selected = value
	#unsetting the flags of the false positives
	for person in fps_to_eliminate:
		obj.loc[(obj.Label == last_fp_object_selected)&(obj.name == "{}.csv".format(person)),'FLAG'] = 'FP'
	#preparing the new list of options after eliminating the false positives
	fp_img_list = [{'label':i, 'value':i} for i in classifier_analysis.get_images_for_prediction(value, obj)]
	if prev_img_list is not None:
		tps_list = [{'label': val, 'value': val} for val in prev_img_list]
	if n_clicks is None or len(fps_to_eliminate)==0:
		return fp_img_list, init_multi_list_values, empty_graph, tps_list
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
	layout = go.Layout(height=700, width=700,yaxis_title= "Average Confidence Score", title = "Change in Avg. Confidence Score for {}".format(last_fp_object_selected),
	 xaxis = dict(showgrid=False, zeroline = False),
	 yaxis = dict(zeroline = False, showgrid=False),
	 plot_bgcolor="#F9F9F9",
	 paper_bgcolor="#F9F9F9"))
	print("Diff in conf score for {} is {}".format(last_fp_object_selected, new_conf_score - old_conf_score))
	#reinitialising fps to eliminate list and highlighted_imgs
	fps_to_eliminate = []
	highlighted_imgs = []
	return fp_img_list, prev_img_list, line_chart, tps_list


prev_test_img = []
@app.callback(Output('person_image', 'children'),
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
	encoded_img= base64.b64encode(open(path,'rb').read())
	prev_test_img = [html.Img(src='data:image/png;base64,{}'.format(encoded_img.decode()),
		  style={'width':'700px', 'height':'700px'})]
	return prev_test_img



@app.callback(Output('image_grid', 'children'),
	[Input('grid_drop_down','value')])
def update_grid(selectedData):
	"""Usage: Analysis 6; Returns: Updates the Training Images grid"""
	if selectedData is None:
		return []
	encoded_img= base64.b64encode(open("MC2/grid/{}_grid.png".format(selectedData),'rb').read())
	return [html.Img(src='data:image/png;base64,{}'.format(encoded_img.decode()),
		   style={'width':'700px', 'height':'700px'})]

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
	plot_bgcolor="#F9F9F9",
	paper_bgcolor="#F9F9F9",
	title = 'Common objects between people',
	xaxis = dict(showticklabels=False),
	yaxis = dict(showticklabels = False),
	height=700,
	width=700,
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

# newlyadded1
@app.callback([Output('BBimage', 'children'),
	Output('ImageFigure', 'figure')],
	[Input('TruePositiveImages','value')])
def updateImage(value):
	global ImageFigureInit,lastImageSelected
	if value is None:
		return [], empty_graph
	print("value", value)
	selected_image  = value

	#path for html images
	directory = 'MC2\MC2-Image-Data_Annotated'
	selected_image_copy = selected_image
	person = selected_image_copy.split('_')[0]
	full_path = os.path.join(directory, person)
	full_path = os.path.join(full_path, selected_image)
	full_path = os.path.splitext(full_path)[0]+'.jpg'

	#path for imshow images
	directory_im = 'MC2\MC2-Image-Data'
	full_path_im = os.path.join(directory_im, person)
	full_path_im = os.path.join(full_path_im, selected_image)
	full_path_im = os.path.splitext(full_path_im)[0]+'.jpg'

	lastImageSelected = selected_image.split('.')[0]
	encoded_img= base64.b64encode(open(full_path,'rb').read())
	image = io.imread(full_path_im)
	image = resize(image, (image.shape[0] // 20, image.shape[0] // 15), anti_aliasing=True)
	ImgFigure = px.imshow(image)
	ImgFigure.update_layout(title = "Reannote here",
	dragmode='drawrect',
	plot_bgcolor="#F9F9F9",
	paper_bgcolor="#F9F9F9",
	newshape=dict(line_color='cyan'))
	return ["Classifier Predictions: " ,html.Img(src='data:image/png;base64,{}'.format(encoded_img.decode()), id=lastImageSelected,
		   style={'width':'600px', 'height':'400px','padding-top':'62px'})], ImgFigure

#newlyadded2
@app.callback(dash.dependencies.Output('dummy', 'children'),
	[dash.dependencies.Input('ImageFigure','relayoutData'),
	dash.dependencies.Input('CorrectedObjects', 'value')])
def updateBoundingBox(relayoutData, value):
	global BoundingBox,lastImageSelected,x_vals,y_vals,width,height,newObject
	if relayoutData is None and value is None:
		return []
	if 'shapes' not in relayoutData:
		return []
	x_vals_cur = list()
	y_vals_cur = list()
	width_cur = list()
	height_cur = list()
	for i in range(0, len(relayoutData['shapes'])):
		x_vals_cur.append(relayoutData['shapes'][i]['x0'])
		y_vals_cur.append(relayoutData['shapes'][i]['y0'])
		width_cur.append(relayoutData['shapes'][i]['x1'] - relayoutData['shapes'][i]['x0'])
		height_cur.append(relayoutData['shapes'][i]['y1'] - relayoutData['shapes'][i]['y0'])
	x_vals_cur = [element * 20 for element in x_vals_cur]
	y_vals_cur = [element * 15 for element in y_vals_cur]
	width_cur = [element * 20 for element in width_cur]
	height_cur = [element * 15 for element in height_cur]
	x_vals = x_vals_cur
	y_vals = y_vals_cur
	width = width_cur
	height =height_cur
	print(x_vals_cur,y_vals_cur,width_cur,height_cur)
	# BoundingBox[0] = (x_vals.min())
	# BoundingBox[1] = (y_vals.min())
	# BoundingBox[2] = (x_vals.max())
	# BoundingBox[3] = (y_vals.max())
	# print("BoundingBox",BoundingBox)
	#DensityVsConfScoreFigNew = classifier_analysis.GetDensityVsConfScoreGraph(objectName=lastImageSelected,xminNew=x_vals.min(), yminNew=y_vals.min(), xmaxNew=x_vals.max(),ymaxNew=y_vals.max())
	#print("x_vals",x_vals)
	#print("y_vals",y_vals)
	#print("x and y min",x_vals.min(),y_vals.min())
	#print("x and y max",x_vals.max(),y_vals.max())
	#return DensityVsConfScoreFigNew

	if value is not None:
		newObject.clear()
		for val in value:
			newObject.append(val)

	return []

#newlyadded3
@app.callback([Output('DensityVsConfScoreFig', 'figure'),
			Output('Exported','displayed')],
	[Input('done-val', 'n_clicks')])
def exportUpdatedBoundingbox(n_clicks):
	global BoundingBox,lastImageSelected,x_vals,y_vals,width,height,newObject
	exportFile = list()
	print(x_vals,y_vals,width,height,newObject)
	for val in range(0, len(x_vals)):
		exportFile.append({'x': x_vals[val], 'y': y_vals[val], 'Width': width[val], 'Width': height[val], 'Label': newObject[val]})
	exportFile = pd.DataFrame(exportFile)
	print("exportFile",exportFile)
	print("lastImageSelected",lastImageSelected)
	lastImageSelectedCur = lastImageSelected + '.csv'
	exportFile.to_csv(lastImageSelectedCur)
    # exportFile.to_csv(full_path, index=False)
	DensityVsConfScoreFigNew = classifier_analysis.GetDensityVsConfScoreGraph()
	if n_clicks is None or n_clicks ==0:
		return DensityVsConfScoreFigNew, False
	print('elim',n_clicks)
	return DensityVsConfScoreFigNew, True


@app.callback([Output('bar_graph','figure'),
	Output('BBSizeVsConfScoreFig', 'figure'),
	Output('obj_distribution', 'figure'),
	Output('score_distribution', 'figure')],
	[Input('bar_graph','clickData'),
	Input('BBSizeVsConfScoreFig', 'clickData'),
	])
def update_graph_colors_and_dist(bar_data, bubble_data):
	"""Usage: Analysis 1; Returns: Updates bar graph and bubble charts' colors on click"""
	#updating all 3  graphs
	global last_bar_idx, last_bubble_idx, last_selected_obj,bar_graph,BBSizeVsConfScoreFig,dist
	new_idx = 0
	if bar_data is None and bubble_data is None:
		return bar_graph, BBSizeVsConfScoreFig, dist2, dist
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
	df = pd.DataFrame({last_selected_obj: label_vs_score[last_selected_obj]})
	dist = ff.create_distplot([df[c] for c in df.columns], df.columns, bin_size=.25)
	dist.update_layout(plot_bgcolor="#F9F9F9",
	paper_bgcolor="#F9F9F9",
	title = "Distribution of Confidence Scores for {}".format(last_selected_obj),
	height=560,
	xaxis_title = "Scores")
	#update distribution across people
	return bar_graph, BBSizeVsConfScoreFig, person_analysis.getDistribution(last_selected_obj), dist

if __name__ == '__main__':
	app.run_server(debug=True)
