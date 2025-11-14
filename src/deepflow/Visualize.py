import matplotlib.pyplot as plt
import numpy as np
from tomlkit import key
from .Network import *
from .Geometry import Area, Bound
from mpl_toolkits.mplot3d import Axes3D
 
class VisualizeTemplate():
    def __init__(self):
        pass
    
    @staticmethod
    def colorplot(X, Y, Data, ax:plt, title=None, cmap='viridis', s=1):
        im = ax.scatter(X, Y, s=s ,c=Data, cmap=cmap, marker = 's')
        ax.set_title(title, fontweight='medium', pad=10, fontsize=13)
        ax.set_xlabel('x', fontstyle='italic', labelpad=0)
        ax.set_ylabel('y', fontstyle='italic', labelpad=0)

        plt.colorbar(im, pad=0.03, shrink=1.2)

        return ax

    @staticmethod
    def lineplot(X, Data, ax:plt, xlabel, ylabel, color='navy'):
        # Main line style
        ax.plot(X, Data, linewidth=2.0, color=color)
        ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.7)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)

        ax.figure.tight_layout()

        return ax

    @staticmethod
    def histplot(Data, ax:plt, title=None, bins='fd'):
        ax.hist(Data, bins, density=True, alpha=0.7, color="steelblue", edgecolor="black")
        ax.set_title(title)

    @staticmethod
    def scatter_points(X, Y, ax, s=1):
        ax.scatter(X, Y, s=s)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        return ax

class Visualizer(VisualizeTemplate):
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.X = data_dict['x']
        self.Y = data_dict['y']
        self.X_min = self.X.min()
        self.X_max = self.X.max()
        self.Y_min = self.Y.min()
        self.Y_max = self.Y.max()
        self.length = self.X_max - self.X_min
        self.width = self.Y_max - self.Y_min

        self.ratio = 5
    
    def plot_data_on_geometry(self, key_cmap_dict, s=10, orientation='vertical', range_x:list=None, range_y:list=None):
        key_cmap_dict, num_plots = self._keycmap_dict_process(key_cmap_dict)
        if orientation == 'vertical':
            fig, axes = plt.subplots(num_plots, 1, figsize=(2.5*self.ratio, 2.5*num_plots))
        elif orientation == 'horizontal':
            fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots*self.ratio, 6))
        
        if num_plots == 1:
            axes = [axes]

        for i, (key,data) in enumerate(key_cmap_dict.items()):
            self.colorplot(self.X,self.Y, self.data_dict[key],axes[i],key,data,s=s)
            axes[i].set_aspect('equal', adjustable='box')
            if range_x:
                axes[i].set_xlim(range_x[0], range_x[1])
            else:
                if self.length < 0.001:
                    axes[i].set_xlim(self.X_min - 0.5*self.width, self.X_max + 0.5*self.width)
                else:
                    axes[i].set_xlim(self.X_min, self.X_max)
            if range_y:
                axes[i].set_ylim(range_y[0], range_y[1])
            else:
                if self.width < 0.001:
                    axes[i].set_ylim(self.Y_min - 0.5*self.length, self.Y_max + 0.5*self.length)
                else:
                    axes[i].set_ylim(self.Y_min, self.Y_max)
        plt.tight_layout()
        plt.show()
        return fig

    def plot_data(self, key_cmap_dict, axis = 'xy'):
        if axis != 'xy':
            key_cmap_dict, num_plots = self._keycmap_dict_process(key_cmap_dict)
            fig, axes = plt.subplots(num_plots, 1)
            if num_plots == 1:
                axes = [axes]
            for i, (key, cmap) in enumerate(key_cmap_dict.items()):
                coord = self.Y if axis == 'y' else self.X
                self.lineplot(coord, self.data_dict[key], axes[i], axis, key)

            plt.tight_layout()
            plt.show()
            return fig
        
        elif axis == 'xy':
            key_cmap_dict, num_plots = self._keycmap_dict_process(key_cmap_dict, 'viridis')
            fig = plt.figure(figsize=(8 * num_plots, 6))

            for i, (key, cmap) in enumerate(key_cmap_dict.items()):
                ax = fig.add_subplot(1, num_plots, i + 1, projection='3d')

                # Create a surface plot
                z = self.data_dict[key]
                ax.scatter(self.X, self.Y, z, c=z, cmap = cmap, marker='.')

                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel(key)
                ax.set_title(f'Surface Plot of {key}')

            plt.tight_layout()
            plt.show()
            return fig

    def plot_data_distribution(self, key:str, bins='fd'):
        fig, ax = plt.subplots()
        ax = plt.subplot()
        self.histplot(self.data_dict[key], ax, f"{key} distribution", bins=bins)
        plt.show()

        return fig

    def plot_loss_curve(self, log_scale=False, linewidth = 0.5):
        fig, ax = plt.subplots()
        ax.plot(self.data_dict['total_loss'], label = 'total_loss', linewidth = linewidth)
        ax.plot(self.data_dict['bc_loss'], label = 'bc_loss', linewidth = linewidth)
        ax.plot(self.data_dict['pde_loss'], label = 'pde_loss', linewidth = linewidth)
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Loss per Iteration")
        ax.legend()  
        plt.show()

        return fig

    @staticmethod
    def _keycmap_dict_process(key_cmap_dict, default:str = 'viridis'):
        if isinstance(key_cmap_dict, dict):
            key_cmap_dict = {key:default if key_cmap_dict[key] is None else key_cmap_dict[key] for key in key_cmap_dict}
        
        if isinstance(key_cmap_dict, (list, tuple)):
            key_cmap_dict = {key:default for key in key_cmap_dict}

        if isinstance(key_cmap_dict, str):
            key_cmap_dict = {key_cmap_dict:default}

        return key_cmap_dict, len(key_cmap_dict)
