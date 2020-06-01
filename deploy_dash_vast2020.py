import dash
import dash_core_components as dcc
import dash_html_components as html
import base64
import plotly.express as px
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
from demographic_analysis import Person
from ClassifierAnalysis import ClassifierAnalysis
import plotly.graph_objects as go

#empty_graph
empty_layout = go.Layout(
plot_bgcolor='rgba(0,0,0,0)',	
xaxis = dict(showticklabels=False, showgrid=False, zeroline = False),
yaxis = dict(showticklabels = False, showgrid=False, zeroline = False),
height=600, width=800,
)
empty_graph = go.Figure()
empty_graph.update_layout(empty_layout)

#Classifier Insight #1 - Confidence Scores of objects
obj = pd.read_csv('Objects.csv')

#Analysis 1 and 2
obj = pd.read_csv('Objects.csv')
classifier_analysis = ClassifierAnalysis(obj)
DensityVsConfScoreFig = classifier_analysis.GetDensityVsConfScoreGraph()
BBSizeVsConfScoreFig = classifier_analysis.GetBBSizeVsConfScoreGraph()

#Classifier Insight #1


uniq_labels = (list(obj.Label.unique())) 
label_vs_score = dict()
for label in uniq_labels:
	label_vs_score[label] = obj[obj.Label==label]['conf_score']
group_by_obj = obj.groupby(by=['Label']).mean()['conf_score']
group_by_obj = group_by_obj.reset_index()
#initial bar selection 
bar_colors = ['lightslategray'] * len(uniq_labels)
bar_colors[1] = 'crimson'

bar_graph = go.Figure(data=[go.Bar(
	x=group_by_obj.Label,
	y=group_by_obj.conf_score,
	marker_color=bar_colors
)])
bar_graph.update_layout(title_text='Average Confidence Scores of Objects',
	xaxis_title= "Objects", 
	height=560,
	yaxis_title="Confidence Scores")
last_selected_obj = bar_graph.data[0].x[1]

#Person Insight #1 - Caption based Connections
person_analysis = Person()
caption_vs_caption = person_analysis.get_caption_to_caption_mapping()
caption_vs_caption_fig = person_analysis.get_caption_2_caption_graph_object(caption_vs_caption, 
	title="Person Connectivity based on captions")

#Peron Insight 2 - Objects Similarity between people
dist_matrix = person_analysis.get_object_similarity_matrix()
heatmap = person_analysis.get_heatmap_fig_for_matrix(dist_matrix)

#external layout 
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#init
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#configuring logo
encoded_logo = base64.b64encode(open('tukl.png','rb').read())
encoded_totem = base64.b64encode(open('totem2.jpg','rb').read())
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
								#Analysis 1
								html.Div([
											dcc.Graph(
											id="BBSizeVsConfScoreFig",
											figure = BBSizeVsConfScoreFig,
										)], className="six columns"),
								#Analysis 2
								html.Div([
											dcc.Graph(
											id="DensityVsConfScoreFig",
											figure = DensityVsConfScoreFig,
										)], className="six columns"),
					], className="row"),
								
					html.Div([	
								##Analysis 2
								html.Div([	
											dcc.Graph(
											id="bar_graph",
											figure = bar_graph,
										)], className="six columns"),
								##Analysis 4
								html.Div([
											dcc.Graph(
											id="conf_score_dist"
										)], className = "six columns")
					], className="row")
				]),	
			#tab2
			dcc.Tab(label='Totem Analysis', children=
				[
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
								#Common objects
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
				])
		])
	])
)

@app.callback(dash.dependencies.Output('similar_objects', 'figure'),
	[dash.dependencies.Input('obj_similarity','clickData')])
def update_similar_object_viz(selectedData):
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
	

@app.callback(dash.dependencies.Output('caption_mapping', 'figure'),
	[dash.dependencies.Input('caption_vs_caption','selectedData')])
def update_obj_mapping_viz(selectedData):
	if selectedData is None:
		return empty_graph.update_layout(title="Common words between persons")
	points = selectedData['points']
	pids = []	
	for point in points:
		person = point['text']
		pids.append(int(person[person.index('#')+1 : person.rindex('|') -1])) 
	return person_analysis.get_sankey_diag_for_ppl(caption_vs_caption, pids)

@app.callback(
	dash.dependencies.Output('bar_graph','figure'),
	[dash.dependencies.Input('bar_graph','clickData')]
	)
def update_bar_color(selectedData):
	if selectedData is None:
		return bar_graph
	bar_idx = selectedData['points'][0]['pointIndex']
	updated_colors = ['lightslategray'] * len(uniq_labels)
	updated_colors[bar_idx ] = 'crimson'
	bar_graph.data[0].marker.color = updated_colors
	return bar_graph
	

@app.callback(
		dash.dependencies.Output('conf_score_dist', 'figure'),
		[dash.dependencies.Input('bar_graph', 'clickData')])
def update_obj_distribution_graph(selectedData):
	global last_selected_obj
	if selectedData is not None:
		last_selected_obj = selectedData['points'][0]['x']
	df = pd.DataFrame({last_selected_obj: label_vs_score[last_selected_obj]})
	fig = ff.create_distplot([df[c] for c in df.columns], df.columns, bin_size=.25)
	fig.update_layout(title = "Distribution of Confidence Scores for {}".format(last_selected_obj),
	height=560,
	xaxis_title = "Scores")
	return fig


if __name__ == '__main__':
	app.run_server(debug=True)