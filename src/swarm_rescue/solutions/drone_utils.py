import numpy as np

def coords_to_indexes(coords, gpsbounds,matrix_size):
    """
    Converts coordinates to indexes in the matrix
    Matrix is a square matrix of size matrix_size
    """

    x,y = coords
    xmin, xmax = gpsbounds
    ymin,ymax = xmin,xmax
    n= matrix_size # nbline; nbcolumn
    i = (y-ymax)/(ymin-ymax)
    j = (x-xmin)/(xmax-xmin)
    return int(i*n),int(j*n)
