import cv2 as cv
from pupil_apriltags import Detector
import argparse
import logging
import os
import numpy as np
import time

def main():
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

    # Initialize variables and logger
    args = get_args()

    # Argument adjustments
    args.camera = 1
    args.families = 'tag16h5'
    
    # Initialize Camera
    cam = cv.VideoCapture(args.camera, cv.CAP_DSHOW)
    while cam is None or not cam.isOpened():
        log.critical('Camera not detected. Press space to check again.')
        if cv.waitKey(1) & 0xFF == ord('q'):
                break

    # Camera Settings
    cam.set(cv.CAP_PROP_AUTO_EXPOSURE, 3)

    # Airtag Detector
    tag_detector = Detector(
        families='tag16h5'
    )

    detection_time = 0
    display_time = 0

    while True:
        try: 
            start_time = time.time()

            ret, image = cam.read()
            if not ret: break
            grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            tags = tag_detector.detect(grayscale_image)

            detection_time = time.time() - start_time
   
            # Outline Tags ==================================================================
            for tag in tags:
                id = tag.tag_id
                center = tag.center
                corners = tag.corners
                if id not in args.tag_id_list: continue
                center = (int(center[0]), int(center[1]))
                corner1 = (int(corners[0][0]), int(corners[0][1]))
                corner2 = (int(corners[1][0]), int(corners[1][1]))
                corner3 = (int(corners[2][0]), int(corners[2][1]))
                corner4 = (int(corners[3][0]), int(corners[3][1]))
                # Draw Elements
                cv.circle(image, (center[0], center[1]), 3, (0,0,255), -1)
                cv.line(image, (corner1[0], corner1[1]), 
                        (corner2[0], corner2[1]), (255,0,0), 2)
                cv.line(image, (corner2[0], corner2[1]), 
                        (corner3[0], corner3[1]), (255,0,0), 2)
                cv.line(image, (corner3[0], corner3[1]), 
                        (corner4[0], corner4[1]), (0,255,0), 2)
                cv.line(image, (corner4[0], corner4[1]), 
                        (corner1[0], corner1[1]), (0,255,0), 2)
                # Place ID # in drawn box
                cv.putText(image, str(id), (center[0] - 10, center[1] - 10),
                            cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                            
            # FPS counter ==================================================================
            # 'Detection' only includes the image capture, manipulation, and detection
            cv.putText(image, f'Detection: {round(detection_time*1000, 3)}ms', (15,30), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            # 'Debug' includes logging functions, rendering, and window updates
            cv.putText(image, f'Display: {round(display_time*1000, 3)}ms', (15,55), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv.putText(image, f'Total: {round((detection_time+display_time)*1000, 3)}ms', (15,80), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
            cv.imshow('AprilTag Detector (Debug)', image)

            display_time = time.time() - start_time - detection_time

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        except cv.error as e:
            log.warning(e)
    cam.release()
    cv.destroyAllWindows()

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

    return args

if __name__ == '__main__':
    main()