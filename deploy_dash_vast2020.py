import dash
import dash_core_components as dcc
import dash_html_components as html
import base64
import plotly.express as px
import numpy as np
import pandas as pd
import plotly.figure_factory as ff

#processed objects from classfier 
obj = pd.read_csv('Objects.csv')
labels = (list(obj.Label.unique())) 
label_vs_score = dict()
for label in labels:
    label_vs_score[label] = obj[obj.Label==label]['conf_score']
group_by_obj = obj.groupby(by=['Label']).mean()['conf_score']
group_by_obj = group_by_obj.reset_index()
#Insight 1
#Conf scores of objects
# treemap = px.treemap(group_by_obj, path=['Label'], values='conf_score',title='Confidence Scores for Objects',
#                   color='conf_score',
#                   color_continuous_scale=px.colors.sequential.YlGnBu,
#                   color_continuous_midpoint=np.average(group_by_obj['conf_score']))


import plotly.graph_objects as go

bar_colors = ['lightslategray'] * len(labels)
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


#external layout 
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#init
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#configuring logo
encoded_img = base64.b64encode(open('tukl.png','rb').read())
#init layout setup
app.layout = html.Div(
    #title and logo
    html.Div([
        html.Div(
            [
            html.Img(src='data:image/png;base64,{}'.format(encoded_img.decode()), style={'float':'right'}),
            html.H1(children='VAST 2020',style={ 'fontSize': 24},className='nine columns'),
            ], className="row"
        ),
        #tabs
        dcc.Tabs([
        #tab1
        dcc.Tab(label='Classifier Analysis', children=[
        #bar_graph
         html.Div([
            dcc.Graph(
                id="bar_graph",
                figure = bar_graph,
                )
            ], className="six columns"),
         #graph2 - distribution of conf_score for each object
        html.Div([
            dcc.Graph(
                id="conf_score_dist"
                )
            ], className = "six columns")
        ]),
        #tab2
        dcc.Tab(label='Demographic Analysis', children=[
        
    ])
    ])
])
)

@app.callback(
	dash.dependencies.Output('bar_graph','figure'),
	[dash.dependencies.Input('bar_graph','clickData')]
	)
def update_color_of_bar_graph(selectedData):
	if selectedData is None:
		return bar_graph
	bar_idx = selectedData['points'][0]['pointIndex']
	updated_colors = ['lightslategray'] * len(labels)
	updated_colors[bar_idx ] = 'crimson'
	bar_graph.data[0].marker.color = updated_colors
	return bar_graph
	

@app.callback(
		dash.dependencies.Output('conf_score_dist', 'figure'),
		[dash.dependencies.Input('bar_graph', 'clickData')])
def update_obj_distribution_graph(selectedData):
	print(selectedData)
	empty_graph = {'data': [],
	'layout': {
	'title': 'Distribution of Confidence Scores'
	}
}
	if selectedData is None:
		return empty_graph 
	points = selectedData['points'][0]
	if 'label' not in points:
		return empty_graph
	selectedObj = selectedData['points'][0]['x']
	idx = labels.index(selectedObj)
	df = pd.DataFrame({selectedObj: label_vs_score[selectedObj]})
	fig = ff.create_distplot([df[c] for c in df.columns], df.columns, bin_size=.25)
	fig.update_layout(title = "Distribution of Confidence Scores for {}".format(selectedObj),
	height=560,
	xaxis_title = "Scores")
	return fig


if __name__ == '__main__':
    app.run_server(debug=True)