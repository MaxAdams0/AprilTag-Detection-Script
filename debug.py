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
    log_file_number = len(os.listdir('logs/'))
    logging.basicConfig(format='[%(asctime)s] [%(levelname)-8s]\n%(message)s\n',
                    level=logging.DEBUG,
                    handlers=[
                        logging.FileHandler(f'logs/debug{log_file_number}.log'),
                        logging.StreamHandler()
                    ])
    
    # Initialize Camera
    cam = cv.VideoCapture(args.camera)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, cam.get(cv.CAP_PROP_FRAME_WIDTH))
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, cam.get(cv.CAP_PROP_FRAME_HEIGHT))

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
        debug_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Detect Apriltags
        tags = at_detector.detect(image)
        debug_image = draw_debug(debug_image, tags)

        # Display debug window
        cv.imshow('Apriltag Detector Debug', debug_image)
        key = cv.waitKey(1)
        if key == 27: # esc key
            break
    # Close
    cam.release()
    cv.destroyAllWindows()

def draw_debug(image, tags):
    log = logging.getLogger()
    for tag in tags:
        id = tag.tag_id
        if id>8: break
        center = tag.center
        corners = tag.corners

        # Tag attributes have diffent type from cv req. arg., conv. to int
        center = (int(center[0]), int(center[1]))
        corner1 = (int(corners[0][0]), int(corners[0][1]))
        corner2 = (int(corners[1][0]), int(corners[1][1]))
        corner3 = (int(corners[2][0]), int(corners[2][1]))
        corner4 = (int(corners[3][0]), int(corners[3][1]))

        # Draw Center
        cv.circle(image, (center[0], center[1]), 5, (0, 0, 255), 2)
        # Note: find which corners corrolate to which
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
                    cv.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255), 2, cv.LINE_AA)
        
        log.debug(f'id:{id}\ncenter:{center}\ncorners:\n{corners}')
        
    return image


if __name__ == '__main__':
    main()