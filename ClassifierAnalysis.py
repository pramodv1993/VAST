import pandas as pd
import numpy as np
import plotly.express as px

# Class for handling the classifier analysis
class ClassifierAnalysis():
    def __init__(self, objects):
            self.objectDB = objects
            
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
        for i, grp in group:
            bb_Size = grp['BoundingBoxSize'].mean()
            conf_mean = grp['conf_score'].mean()
            var_conf = grp['conf_score'].var()
            avg_bb_size.append(bb_Size)
            conf_score_mean.append(conf_mean)
            conf_score_var.append(var_conf)
    
        fig = px.scatter(x=conf_score_mean, y=conf_score_var, size=avg_bb_size)
        fig.update_layout(xaxis_title="Conf_Score mean", yaxis_title="Conf_Score Variance")
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
        
        fig = px.scatter(x= df['average_density'], y=df['average_conf_score'])
        fig.update_layout(xaxis_title="Average Density of Objects", yaxis_title="Conf_Score Mean")
        fig.update_xaxes(range=[0.00001, 0.0003])
        return fig
 