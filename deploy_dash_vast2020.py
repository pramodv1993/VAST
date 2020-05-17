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
treemap = px.treemap(group_by_obj, path=['Label'], values='conf_score',title='Confidence Scores for Objects',
                  color='conf_score',
                  color_continuous_scale=px.colors.sequential.YlGnBu,
                  color_continuous_midpoint=np.average(group_by_obj['conf_score']))
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
        #treemap
         html.Div([
            dcc.Graph(
                id="treemap",
                figure = treemap,
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
    dash.dependencies.Output('conf_score_dist', 'figure'),
    [dash.dependencies.Input('treemap', 'clickData')])
def update_obj_distribution_graph(selectedData):
    empty_graph = {
                        'data': [
                        ],
                        'layout': {
                            'title': 'Distribution of Confidence Scores'
                        }
                    }
    if selectedData is None:
        return empty_graph 
    points = selectedData['points'][0]
    if 'label' not in points:
        return empty_graph
    selectedObj = selectedData['points'][0]['label']
    idx = labels.index(selectedObj)
    df = pd.DataFrame({selectedObj: label_vs_score[selectedObj]})
    fig = ff.create_distplot([df[c] for c in df.columns], df.columns, bin_size=.25)
    fig.update_layout(title = "Distribution of Confidence Scores for {}".format(selectedObj), xaxis_title = "Scores")
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)