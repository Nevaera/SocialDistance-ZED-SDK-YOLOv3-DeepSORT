# Imports
import numpy as np # For Arrays

class Person:
    '''
        Stores location information for tracked person to compute distance to other people
    '''
    cx = -1
    cy = -1
    x = -1.0
    y = -1.0
    z = -1.0
    bbox = None
    dsid = -1
    d = -1
    def __init__(self, cx, cy, x, y, z, bbox, dsid, d):
        self.cx = cx
        self.cy = cy
        self.x = x
        self.y = y
        self.z = z
        self.bbox = bbox
        self.dsid = dsid
        self.d = d

def load_image_into_numpy_array(image):
    '''
        Loads an image from the ZED into a uint numpy array for use with image processing
    '''
    ar = image.get_data()
    ar = ar[:, :, 0:3]
    (im_height, im_width, channels) = image.get_data().shape
    return np.array(ar).reshape((im_height, im_width, 3)).astype(np.uint8)

def load_depth_into_numpy_array(depth):
    '''
        Loads an point cloud from the ZED into a float numpy array for use with depth processing
    '''
    ar = depth.get_data()
    ar = ar[:, :, 0:4]
    (im_height, im_width, channels) = depth.get_data().shape
    return np.array(ar).reshape((im_height, im_width, channels)).astype(np.float32)

def get_center(min, max):
    return int((min + max) * 0.5)

def compute_relative_distance(e_a, e_b):
    a = np.array((e_a.x, e_a.y, e_a.z))
    b = np.array((e_b.x, e_b.y, e_b.z))
    distance = np.linalg.norm(a-b)
    return distance

