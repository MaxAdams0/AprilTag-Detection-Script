import cv2 as cv
from pupil_apriltags import Detector
import copy
import argparse
import logging
import os

def get_args():
    # Note: defaults according to documentation
    # https://pupil-apriltags.readthedocs.io/en/stable/api.html
    parser = argparse.ArgumentParser()
    # Camera args
    parser.add_argument('--camera', type=int, default=0)
    # Detection args
    parser.add_argument('--families', type=str, default='tag36h11')
    parser.add_argument('--nthreads', type=int, default=1)
    parser.add_argument('--quad_decimate', type=float, default=2.0)
    parser.add_argument('--quad_sigma', type=float, default=0.0)
    parser.add_argument('--refine_edges', type=int, default=1)
    parser.add_argument('--decode_sharpening', type=float, default=0.25)
    parser.add_argument('--debug', type=int, default=0)

    args = parser.parse_args()
    return args

def main():
    # Initializing variables
    args = get_args()
    # Create log file and setup logger
    log_file_number = len(os.listdir('logs/'))
    logging.basicConfig(format='[%(asctime)s] [%(levelname)-8s]\n%(message)s\n',
                    level=logging.DEBUG,
                    handlers=[
                        logging.FileHandler(f'logs/debug{log_file_number}.log'),
                        logging.StreamHandler()
                    ])
    
    # Initialize Camera
    cam = cv.VideoCapture(args.camera)

    # Argument adjustments
    args.families = 'tag16h5'
    args.quad_decimate = 0.0
    args.quad_sigma = 5.0

    # Airtag Detector
    families = args.families
    nthreads = args.nthreads
    quad_decimate = args.quad_decimate
    quad_sigma = args.quad_sigma
    refine_edges = args.refine_edges
    decode_sharpening = args.decode_sharpening
    debug = args.debug

    at_detector = Detector(
        families=families,
        nthreads=nthreads,
        quad_decimate=quad_decimate,
        quad_sigma=quad_sigma,
        refine_edges=refine_edges,
        decode_sharpening=decode_sharpening,
        debug=debug,
    )

    while True:
        # Get camera data & convert to grayscale
        ret, image = cam.read()
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Detect Apriltags & run logging function
        logger(at_detector.detect(image))

        key = cv.waitKey(1)
        if key == 27: # esc key
            break
    # Close
    cam.release()

def logger(tags):
    log = logging.getLogger()
    for tag in tags:
        id = tag.tag_id
        if id>8: break
        center = tag.center
        corners = tag.corners
        
        log.debug(f'id:{id}\ncenter:{center}\ncorners:\n{corners}')


if __name__ == '__main__':
    main()