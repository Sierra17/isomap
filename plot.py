import numpy as np
import plotly.express as px
from PIL import Image


def scatterplot(X, title=None):

    """
    Plots scatterplot of data using Plotly in either 2D or 3D.

    Inputs
    ----------
    X : ndarray of shape (n_samples, d)
        The input data.

    title: str, optional, default=None
        The title of the plot. Set to "2/3D Embedding from ISOMAP" if None.

    Outputs
    -------
    fig : plotly.graph_objects.Figure
        The generated Plotly figure.
    """

    n, d = X.shape

    assert d==2 or d==3, "This function only accepts embeddings in 2D or 3D - try again."

    if title is None:
        title = f"{d}D Embedding from ISOMAP"

    if d == 2:
        fig = px.scatter(x=X[:, 0], y=X[:, 1], title=title)
    elif d == 3:
        fig = px.scatter_3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], title=title)

    fig.update_traces(marker_size=5)

    return fig


def image_annotations(fig, X, image_paths, image_idx, sizex=5, sizey=5, subset=False):

    """
    Mutates Plotly figure object by adding image annotations.

    Inputs
    ----------
    fig : plotly.graph_objects.Figure
        The existing Plotly figure.

    X : ndarray of shape (n_samples, d)
        The input data.
    
    image_paths : list of str, optional (default=None)
        List of image file paths corresponding to selected observations.

    image_idx : list of int, optional (default=None)
        The indices of the observations to overlay with corresponding images (only for 2D).

    sizex : int, optional (default=5)
        The width of each image annotation.

    sizey : int, optional (default=5)
        The length of each image annotation.

    subset : bool, optional (default=False)
        Indicator whether ISOMPA was performed on a subset of the original data.

    Outputs
    -------
    fig : plotly.graph_objects.Figure
        The Plotly figure with image annotations.
    """

    if subset:
        order_idx = np.argsort(image_idx)
    else:
        order_idx = image_idx

    for i, j, image_path in zip(image_idx, order_idx, image_paths):
        image = Image.open(image_path)
        fig.add_layout_image(
            dict(
                source=image,
                x=X[j, 0],
                y=X[j, 1],
                xref='x',
                yref='y',
                sizex=sizex,
                sizey=sizey,
                xanchor='center',
                yanchor='middle'
            )
        )
    
    return None

