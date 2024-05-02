
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from mpl_toolkits import mplot3d
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker



class Graphs:
    def __init__(self, dataset,actual,predicted):
        self.dataset = dataset
        self.actual = actual
        self.predicted = predicted
  
    #test the data and get the accuracy 
    import matplotlib.patches as mpatches

    def scatterPlot(self, title):
        z = self.dataset["Protocol"]
        x = self.dataset["Length"]
        y = self.dataset["Time"]
        websitesActual = self.actual
        websitesPredicted = self.predicted
        color_map = {'Linkedin': 'red', 'Blackboard': 'blue', 'ChatGPT': 'green'}

        # --- Create actual classified packets plot ---
        figActual = plt.figure(figsize=(16, 9))
        ax = plt.axes(projection="3d")

        # Add x, y gridlines
        ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.3, alpha=0.2)

        # Create colors based on website
        colors = [color_map.get(name, 'gray') for name in websitesActual]

        # Create scatter plot
        ax.scatter3D(x, y, z, c=colors, marker='o')

        plt.title("Actual Classified Packets")
        ax.set_xlabel('Length Of Packet', fontweight='bold')
        ax.set_ylabel('Packet Transmission Time', fontweight='bold')
        ax.set_zlabel('Protocol Type', fontweight='bold')

        # --- Create legend for actual classified packets ---
        handles = [mpatches.Patch(color=color_map[website]) for website in color_map]
        labels = color_map.keys()
        ax.legend(handles, labels, title="Website")  # Add legend to the actual plot

        # --- Create predicted packets plot ---
        figPredicted = plt.figure(figsize=(16, 9))
        ax = plt.axes(projection="3d")

        # Add x, y gridlines
        ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.3, alpha=0.2)

        colors = [color_map.get(name, 'gray') for name in websitesPredicted]

        # Create scatter plot  
        ax.scatter3D(x, y, z, c=colors, marker='o')

        plt.title("Predicted Packets using " + title)
        ax.set_xlabel('Length of Packet', fontweight='bold')
        ax.set_ylabel('Packet Transmission Time', fontweight='bold')
        ax.set_zlabel('Protocol Type', fontweight='bold')

        # --- Create legend for predicted packets ---
        handles = [mpatches.Patch(color=color_map[website]) for website in color_map]
        labels = color_map.keys()
        ax.legend(handles, labels, title="Website")  # Add legend to predicted plot

        return figActual, figPredicted

    def confusionMatrix(self, title):
        cm = metrics.confusion_matrix(self.actual, self.predicted)
        ax= plt.subplot()
        sns.heatmap(cm, annot=True,cmap='gist_gray', fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
        ax.set_xlabel('Predicted');ax.set_ylabel('Actual'); 
        ax.set_title(title); 
        ax.xaxis.set_ticklabels(np.unique(self.predicted)); ax.yaxis.set_ticklabels(np.unique(self.predicted))
        return 






