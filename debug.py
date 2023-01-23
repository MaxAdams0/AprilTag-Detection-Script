import cv2 as cv
from pupil_apriltags import Detector
import copy
import argparse
import logging
import os
import keyboard as kb

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
    # The FRC stage has 8 total Apriltags, 0-7
    parser.add_argument('--tag_maxim', type=int, default=7)

    args = parser.parse_args()
    return args

def main():
    # Initialize variables
    args = get_args()
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
    log = logging.getLogger()
    
    # Initialize Camera
    cam = cv.VideoCapture(args.camera)
    while cam is None or not cam.isOpened():
        log.critical('Camera not detected. Press space to check again.')
        kb.wait('space')

    # Argument adjustments
    args.families = 'tag16h5'
    args.quad_decimate = 0.0
    args.quad_sigma = 5.0

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

    while True:
        # Get camera data & convert to grayscale
        ret, image = cam.read()
        debug_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Detect Apriltags & run logging function
        tags = tag_detector.detect(image)
        tag_logger(tags)
        debug_image = draw_debug(debug_image, tags)

        # Display debug window
        cv.imshow('AprilTag Detector (Debug)', debug_image)
        # I have no idea if this works
        key = cv.waitKey(1)
        if key == 27: # esc key
            break
    cam.release()
    cv.destroyAllWindows()

def draw_debug(image, tags):
    args = get_args()
    for tag in tags:
        # Tag attributes
        id = tag.tag_id
        center = tag.center
        corners = tag.corners
        if id>args.tag_maxim: break

        # Tag attributes have diffent type from cv req. arg., conv. to int
        center = (int(center[0]), int(center[1]))
        corner1 = (int(corners[0][0]), int(corners[0][1]))
        corner2 = (int(corners[1][0]), int(corners[1][1]))
        corner3 = (int(corners[2][0]), int(corners[2][1]))
        corner4 = (int(corners[3][0]), int(corners[3][1]))

        # Draw Center
        cv.circle(image, (center[0], center[1]), 5, (0, 0, 255), 2)
        # To-do: find which corners corrolate to which
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
                    cv.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255), 2,)
        
    return image

def tag_logger(tags):
    args = get_args()
    log = logging.getLogger()
    for tag in tags:
        # Tag attributes
        id = tag.tag_id
        center = tag.center
        corners = tag.corners
        if id>args.tag_maxim: break
        
        log.info(f'\nid:{id}\ncenter:{center}\ncorners:\n{corners}\n')


if __name__ == '__main__':
    main()