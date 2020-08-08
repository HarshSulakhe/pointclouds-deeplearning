import plotly.graph_objs as go
import plotly.express as ps
import numpy as np
import math
import random
import os

def read_off(filepath):
    """
    The description of a .off file can be found at https://shape.cs.princeton.edu/benchmark/documentation/off_format.html.
    This function simply parses the text of the file to output faces and vertices of a point cloud.
    """
    f = open(filepath,'r')
    ## Verify if File header is a valid OFF header
    if "OFF" != f.readline().strip():
        print("Not a valid OFF file, returning None")
        return None,None
    # print(f.readline().strip() == "OFF")
    ## Now parse number of faces and vertices
    values = f.readline().strip()
    n_verts,n_faces,_ = tuple([int(i) for i in values.split(' ')])
    ## Store values for vertices and faces (vertrices first)
    verts = [[float(vert) for vert in f.readline().strip().split(' ')] for i in range(n_verts)]
    faces = [[float(face) for face in f.readline().strip().split(' ')][1:] for i in range(n_faces)]

    # print("Number of vertices = " + str(len(verts)),"Number of faces = " + str(len(faces)))
    f.close()
    return verts,faces


def visualize_rotate(data):
    """
    Uses plotly's functionality to plot an interactive 3D image.
    Code taken from  https://github.com/nikitakaraevv/pointnet/blob/master/nbs/PointNetClass.ipynb .
    Data has to be a type of plotly.graph_objs Object.
    """
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames=[]

    def rotate_z(x, y, z, theta):
        w = x+1j*y
        return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))
    fig = go.Figure(data=data,
                    layout=go.Layout(
                        updatemenus=[dict(type='buttons',
                                    showactive=False,
                                    y=1,
                                    x=0.8,
                                    xanchor='left',
                                    yanchor='bottom',
                                    pad=dict(t=45, r=10),
                                    buttons=[dict(label='Play',
                                                    method='animate',
                                                    args=[None, dict(frame=dict(duration=50, redraw=True),
                                                                    transition=dict(duration=0),
                                                                    fromcurrent=True,
                                                                    mode='immediate'
                                                                    )]
                                                    )
                                            ]
                                        )
                                    ]
                                    ),
                    frames=frames
                    )

    return fig

def pcshow(xs,ys,zs):
    """
    Function used to plot an interactive point cloud object.
    """

    data=[go.Scatter3d(x=xs, y=ys, z=zs,
                                   mode='markers')]
    fig = visualize_rotate(data)
    fig.update_traces(marker=dict(size=2,
                      line=dict(width=2,
                      color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.show()

def normalize_pointcloud(pointcloud):
    """
    Pointcloud data will now have zero mean and will be centered around the origin, normalized to a unit sphere.
    """

    pointcloud -= pointcloud.mean(axis = 0)
    pointcloud /= np.max(np.linalg.norm(pointcloud,axis = 1))

    return pointcloud
