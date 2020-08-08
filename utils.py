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
    faces = [[float(face) for face in f.readline().strip().split(' ')] for i in range(n_faces)]

    # print("Number of vertices = " + str(len(verts)),"Number of faces = " + str(len(faces)))
    f.close()
    return verts,faces
