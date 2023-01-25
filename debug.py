import cv2 as cv
from pupil_apriltags import Detector
import argparse
import logging
import os
import keyboard as kb
import numpy as np
import time

def main_debug(args):
    detection_time = 0
    display_time = 0

    while True:
        start_time = time.time()

        ret, image = cam.read()
        debug_image = np.copy(image)
        grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Detect Apriltags & run logging function
        tags = tag_detector.detect(grayscale_image)

        detection_time = time.time() - start_time

        tag_logger(tags, args)
        debug_image = draw_tags(debug_image, tags, args)
        debug_image = draw_fps(debug_image, detection_time, display_time)
        
        # Adds videos together to display side-by-side on window; Axis, 1=hor. 0=vert.
        # Both arrays need to be of the same type
        grayscale_image = cv.cvtColor(grayscale_image, cv.COLOR_GRAY2BGR)
        both = np.concatenate((debug_image, grayscale_image), axis=1)

        # Display debug window
        cv.imshow('AprilTag Detector (Debug)', both)

        display_time = time.time() - start_time - detection_time

        # I have no idea if this works
        key = cv.waitKey(1)
        if key == 27: # esc key
            break
    cam.release()
    cv.destroyAllWindows()

def draw_tags(image, tags, args):
    args, log = get_args()
    for tag in tags:
        # Tag attributes
        id = tag.tag_id
        center = tag.center
        corners = tag.corners
        if id not in args.tag_id_list: break

        # Tag attributes have diffent type from cv req. arg., conv. to int
        center = (int(center[0]), int(center[1]))
        corner1 = (int(corners[0][0]), int(corners[0][1]))
        corner2 = (int(corners[1][0]), int(corners[1][1]))
        corner3 = (int(corners[2][0]), int(corners[2][1]))
        corner4 = (int(corners[3][0]), int(corners[3][1]))

        # Draw Center
        cv.circle(image, (center[0], center[1]), 5, (0, 0, 255), 2)
        # Draw Box
        cv.line(image, (corner1[0], corner1[1]),
                (corner2[0], corner2[1]), (255, 0, 0), 2)
        cv.line(image, (corner2[0], corner2[1]),
                (corner3[0], corner3[1]), (255, 0, 0), 2)
        cv.line(image, (corner3[0], corner3[1]),
                (corner4[0], corner4[1]), (0, 255, 0), 2)
        cv.line(image, (corner4[0], corner4[1]),
                (corner1[0], corner1[1]), (0, 255, 0), 2)

        # Place ID # in drawn box
        cv.putText(image, str(id), (center[0] - 10, center[1] - 10),
                    cv.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255), 2)

    return image

def draw_fps(image, detection_time, display_time):
    # FPS counter ==============================================================
    FPS_color = np.ndarray.tolist(image[0,0])
    # Invert pixel colors
    for channel in range(len(FPS_color)):
        FPS_color[channel] = 255-FPS_color[channel]

    # Convert frame time from seconds to milliseconds
    detection_time = round(detection_time*1000, 3)
    display_time = round(display_time*1000, 3)
    # 'Detection' only includes the image capture, manipulation, and detection
    cv.putText(image, f'Detection: {detection_time}ms', (10,30), 
                cv.FONT_HERSHEY_SIMPLEX, 0.6, FPS_color, 2)
    # 'Debug' includes logging functions, rendering, and window updates
    cv.putText(image, f'Debug: {display_time}ms', (10,55), 
                cv.FONT_HERSHEY_SIMPLEX, 0.6, FPS_color, 2)
        
    return image

def tag_logger(tags, args):
    for tag in tags:
        # Tag attributes
        id = tag.tag_id
        center = tag.center
        corners = tag.corners
        if id not in args.tag_id_list: break
        
        log.info(f'\nid:{id}\ncenter:{center}\ncorners:\n{corners}\n')

def get_args():
    # Note: defaults according to documentation
    # https://pupil-apriltags.readthedocs.io/en/stable/api.html
    parser = argparse.ArgumentParser()
    # Camera variables ===================================================
    parser.add_argument('--camera', type=int, default=0)
    # Detection variables ================================================
    parser.add_argument('--families', type=str, default='tag36h11')
    parser.add_argument('--nthreads', type=int, default=1)
    parser.add_argument('--quad_decimate', type=float, default=2.0)
    parser.add_argument('--quad_sigma', type=float, default=0.0)
    parser.add_argument('--refine_edges', type=int, default=1)
    parser.add_argument('--decode_sharpening', type=float, default=0.25)
    parser.add_argument('--debug', type=int, default=0)
    # FRC stage variables ================================================
    # The FRC stage has 8 total Apriltags, 1-8
    parser.add_argument('--tag_id_list', type=list, default=[1,2,3,4,5,6,7,8])

    args = parser.parse_args()

    # Create log file and setup logger
    if not os.path.exists('logs/'):
        os.mkdir('logs/')
    log_file_number = len(os.listdir('logs/'))
    logging.basicConfig(format='[%(asctime)s] [%(levelname)-8s] %(message)s',
                    level=logging.DEBUG,
                    handlers=[
                        logging.FileHandler(f'logs/debug{log_file_number}.log'),
                        logging.StreamHandler()
                    ])

    return args, logging.getLogger()

if __name__ == '__main__':
    # Initialize variables and logger
    args, log = get_args()

    # Argument adjustments
    args.camera = 1
    args.families = 'tag16h5'
    args.quad_decimate = 0.0
    args.quad_sigma = 5.0
    args.decode_sharpening = 5
    
    # Initialize Camera
    cam = cv.VideoCapture(args.camera)
    while cam is None or not cam.isOpened():
        log.critical('Camera not detected. Press space to check again.')
        kb.wait('space')

    # Airtag Detector
    tag_detector = Detector(
        families=args.families,
        nthreads=args.nthreads,
        quad_decimate=args.quad_decimate,
        quad_sigma=args.quad_sigma,
        refine_edges=args.refine_edges,
        decode_sharpening=args.decode_sharpening,
        debug=args.debug
    )
    
    main_debug(args)