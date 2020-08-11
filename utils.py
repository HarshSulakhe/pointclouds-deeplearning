import plotly.graph_objs as go
import plotly.express as ps
import numpy as np
import math
import random
import os

class PointSampler(object):
    """
    Since original point clouds are not visually representative of the corresponding object, points are sampled randomly on each of the faces of the mesh.
    The following code has not been written by me, due credit goes to https://github.com/nikitakaraevv
    """

    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * ( side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0)**0.5

    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t-s)*pt2[i] + (1-t)*pt3[i]
        return (f(0), f(1), f(2))


    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = (self.triangle_area(verts[faces[i][0]],
                                           verts[faces[i][1]],
                                           verts[faces[i][2]]))

        sampled_faces = (random.choices(faces,
                                      weights=areas,
                                      cum_weights=None,
                                      k=self.output_size))

        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = (self.sample_point(verts[sampled_faces[i][0]],
                                                   verts[sampled_faces[i][1]],
                                                   verts[sampled_faces[i][2]]))

        return sampled_points


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
    faces = [[int(face) for face in f.readline().strip().split(' ')][1:] for i in range(n_faces)]

    # print("Number of vertices = " + str(len(verts)),"Number of faces = " + str(len(faces)))
    f.close()
    return verts,faces

def read_new(filepath):

    with open(filepath,'r') as f:
        x = f.readlines()
        h = [[float(y) for y in n ] for n in [l.strip().split(' ') for l in x]]
        return (np.array(h))

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
    pointcloud is of shape (N,3) where N is the number of vertices
    """

    pointcloud -= pointcloud.mean(axis = 0)
    pointcloud /= np.max(np.linalg.norm(pointcloud,axis = 1))

    return pointcloud

def rotate_pointcloud(pointcloud):
    """
    Function to randomly rotate a pointcloud by an angle theta about the z axis, this is useful to make a deep learning model view invariant.
    Explanation can be found at https://en.wikipedia.org/wiki/Rotation_matrix#Basic_rotations.
    pointcloud is of shape (N,3) where N is the number of vertices.
    """
    theta = 2*math.pi*np.random.random()
    rotation_matrix = np.array([[math.cos(theta),-math.sin(theta),0],[math.sin(theta),math.cos(theta),0],[0,0,1]])

    rotated_pointcloud = np.dot(rotation_matrix,pointcloud.T).T ## Shape N,3

    return rotated_pointcloud

def noise_pointcloud(pointcloud):
    """
    To make a deep learning model robust to variances in distributions of input data, a slight noise is added to the original pointcloud.
    """

    noise = np.random.normal(0,0.05,size = (pointcloud.shape))

    noisy_pointcloud = pointcloud + noise
    return noisy_pointcloud
