'''
    # This code has been adapted from: https://github.com/theAIGuysCode/yolov3_deepsort
    # - For use with YOLO and Tensorflow2 for object identification
    # - For use with DeepSORT for object tracking
    #
    # and use of the ZED API: https://www.stereolabs.com/docs/object-detection/using-object-detection/
    # - For use with Stereolabs ZED camera point cloud
    #
'''
#Imports added to support the addition of the ZED
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Comment out to unsuppress TensorFlow's Excessive debug messages
#from os import system, name
from threading import Lock, Thread
from time import sleep

import time, random
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import math
import matplotlib.pyplot as plt
import statistics
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from PIL import Image

import pyzed.sl as sl

#import library # Contains functions to perform matrix-related operations for the ZED images and point cloud
# Function Imports
from library import load_image_into_numpy_array
from library import load_depth_into_numpy_array
from library import get_center
from library import compute_relative_distance
from library import Person

flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_boolean('depth', False, 'centerpoint or median')
flags.DEFINE_integer('size', 416, 'resize images to')
#flags.DEFINE_string('video', './data/video/test.mp4',
#                    'path to video file or number for webcam)')
#flags.DEFINE_string('output', None, 'path to output video')
#flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_float('distance', 2.0, 'the social distance [in metres /m]')

#Stereolabs ZED-related global variables
lock = Lock()   
width = 1280#2560#704 This is the resolution of the ZED Camera
height = 720#416
image_np_global = np.zeros([width, height, 3], dtype=np.uint8) # Global numpy array for ZED left image
depth_np_global = np.zeros([width, height, 4], dtype=np.float) # Global numpy array for ZED depth map
exit_signal = False                         # Global variable for exiting
new_data = False                            # Global variable for indicating if new data is available from ZED

def zed_capture_thread_func():
    global image_np_global, depth_np_global, exit_signal, new_data

    print(">>> ZED Start...")
    zed = sl.Camera()
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720#VGA#HD720
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.svo_real_time_mode = False

    # Open the camera
    err = zed.open(init_params)
    print(err)
    while err != sl.ERROR_CODE.SUCCESS:
        err = zed.open(init_params)
        print(">>> " + str(err))
        sleep(1)

    image_mat = sl.Mat() # Image Matrix
    depth_mat = sl.Mat() # Depth Matrix
    runtime_parameters = sl.RuntimeParameters()
    print(">>> ZED Start... Done!")

    while not exit_signal:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Get Left Image
            zed.retrieve_image(image_mat, sl.VIEW.LEFT)            
            # Get Depth Image
            zed.retrieve_measure(depth_mat, sl.MEASURE.XYZRGBA)
            
            lock.acquire()
            image_np_global = load_image_into_numpy_array(image_mat)
            depth_np_global = load_depth_into_numpy_array(depth_mat)
            new_data = True
            lock.release()
    print("<<< ZED Shutdown...")
    zed.close()
    print("<<< ZED Shutdown... Done!")


def main(_argv):
    global image_np_global, depth_np_global, exit_signal, new_data
    
    # Definition of the parameters
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 1.0
    
    #initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    print(" >>> Load Weights...")
    yolo.load_weights(FLAGS.weights)
    #logging.info('weights loaded')
    print(" >>> Load Weights... Done!")
    print(" >>> Load Classes...")
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    #logging.info('classes loaded')
    print(" >>> Load Classes... Done!")
    fps = 0.0
    social_distance = FLAGS.distance
    '''    
        # Main Loop
    '''
    print("     >>> Program Start... [Press 'Q/q' to quit]")
    while True:
        t1 = time.time()
        # Colors in BGR format
        color_white = (255, 255, 255)
        color_red = (0, 0, 255)    
        color_green = (0, 255, 0) 
        if new_data:
            # Get images from ZED capture thread
            lock.acquire()
            image_np = np.copy(image_np_global)
            depth_np = np.copy(depth_np_global)
            new_data = False
            lock.release()    
            # Convert to OpenCV format
            image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB) 
            image = tf.expand_dims(image, 0)
            image = transform_images(image, FLAGS.size)

            # Detect Items in Image 
            boxes, scores, classes, nums = yolo.predict(image)
            classes = classes[0]
            names = []
            #print("len classes " + str(len(classes)))
            for i in range(len(classes)):
                #print("int classes " + str(int(classes[i])))
                names.append(class_names[int(classes[i])])
            names = np.array(names)
            converted_boxes = convert_boxes(image_np, boxes[0])
            features = encoder(image_np, converted_boxes)  
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features)]

            #initialize color map
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            # run non-maxima suppresion
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            ### UNCOMMENT BELOW IF YOU WANT CONSTANTLY CHANGING YOLO DETECTIONS TO BE SHOWN ON SCREEN
            #for det in detections:
            #    bbox = det.to_tlbr() 
            #    name = det.get_class()
            #    cv2.rectangle(image_np,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
            #    cv2.putText(image_np, name, (int(bbox[0]), int(bbox[1]-10)), 0, 0.75, (255,255,255),2)

            people = [] # List of locations of all tracked people in the frame

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                class_name = track.get_class()

                # Get only people
                if not str(class_name) == "person":
                    continue
                else:
                    xmin = int(bbox[0]) # Y first then X
                    ymin = int(bbox[1])
                    xmax = int(bbox[2])
                    ymax = int(bbox[3])
                    
                    # Clamp to Image frame (if boundingbox falls outside of frame)
                    #if ymin < 0: ymin = 0
                    #if xmin < 0: xmin = 0
                    #if ymax > height: ymax = height-1
                    #if xmax > width: xmax = width-1

                    # Centerpoint depth
                    xc = get_center(xmin, xmax) #int((xmax + xmin) * 0.5)
                    yc = get_center(ymin, ymax) #int((ymax + ymin) * 0.5)
                   
                    if FLAGS.depth:
                        # Get depth of center-point from cloud
                        x = depth_np[yc, xc, 0]
                        y = depth_np[yc, xc, 1]
                        z = depth_np[yc, xc, 2]

                        if not np.isnan(z) and not np.isinf(z):
                            cv2.circle(image_np, (xc, yc), 5, (0, 255, 0), thickness=cv2.FILLED) # Green Dot
                            cv2.circle(depth_np, (xc, yc), 5, (0, 255, 0), thickness=cv2.FILLED) # Green Dot
                            distance = math.sqrt(x * x + y * y + z * z) # Compute distance from camera 
                            people.append(Person(xc, yc, x, y, z, bbox, track.track_id, distance))
                        else:
                            cv2.circle(image_np, (xc, yc), 5, (0, 0, 255), thickness=cv2.FILLED) # Red Dot
                            cv2.circle(depth_np, (xc, yc), 5, (0, 0, 255), thickness=cv2.FILLED) # Red Dot
                            continue
                    else:
                        # Median depth
                        x_vect = []
                        y_vect = []
                        z_vect = []
                        for j_ in range(ymin, ymax):
                            if np.mod(j_, 25) == 0:
                                for i_ in range(xmin, xmax):
                                    if np.mod(i_, 10) == 0:
                                        try:
                                            z = depth_np[j_, i_, 2]
                                            if not np.isnan(z) and not np.isinf(z):
                                                x_vect.append(depth_np[j_, i_, 0])
                                                y_vect.append(depth_np[j_, i_, 1])
                                                z_vect.append(z)
                                                cv2.circle(depth_np, (i_, j_), 2, color_green, thickness=cv2.FILLED) # Green Dot
                                            else:
                                                cv2.circle(depth_np, (i_, j_), 2, color_red, thickness=cv2.FILLED) # Red Dot
                                        except IndexError:
                                            continue
                        distance = -1.0
                        if len(x_vect) > 0:
                            x = statistics.median(x_vect)
                            y = statistics.median(y_vect)
                            z = statistics.median(z_vect)
                            distance = math.sqrt(x * x + y * y + z * z) # Compute distance from camera 
                            people.append(Person(xc, yc, x, y, z, bbox, track.track_id, distance))                                        
                
            for person in people:
                # Draw Bounding Boxes
                color = colors[int(person.dsid) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(image_np, (int(person.bbox[0]), int(person.bbox[1])), (int(person.bbox[2]), int(person.bbox[3])), color, 2)
                cv2.rectangle(depth_np, (int(person.bbox[0]), int(person.bbox[1])), (int(person.bbox[2]), int(person.bbox[3])), color, 2)
                cv2.rectangle(image_np, (int(person.bbox[0]), int(person.bbox[1]-30)), (int(person.bbox[0])+(len(class_name)+len(str(person.dsid)))*17, int(person.bbox[1])), color, -1)
                cv2.putText(image_np, "person " + str(person.dsid),(int(person.bbox[0]), int(person.bbox[1]-10)),0, 0.75, color_white,2)
                # Display distance to other people
                for person2 in people:
                    if not person.dsid == person2.dsid:
                        #Check if the two people are socially distanced
                        d = round(compute_relative_distance(person, person2), 2)
                        if d >= social_distance:
                            color = color_green
                        else:
                            color = color_red                        
                        cv2.line(image_np, (person.cx, person.cy), (person2.cx, person2.cy), color_white, 2)
                        cx = get_center(person.cx, person2.cx)
                        cy = get_center(person.cy, person2.cy)                        
                        cv2.putText(image_np, str(round(d, 2)) + "m", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            
            # print fps on screen 
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            cv2.putText(image_np, "FPS: {:.2f}".format(fps), (0, 30),
            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color_red, 2)
            
            # show output
            cv2.imshow('Left Image', image_np)
            cv2.imshow('Depth Image', depth_np)
                
        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            print("     <<< Exiting...")
            lock.acquire()
            exit_signal = True
            lock.release()
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        capture_thread = Thread(target=zed_capture_thread_func)
        capture_thread.start()
        app.run(main)
    except SystemExit:
        pass
