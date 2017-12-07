import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import pickle
import argparse
import time
from utils import *
from parameters import *
from scipy.ndimage.measurements import label


def detect(img, dist_pickle):
    draw_image = np.copy(img)
    # the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    img = img.astype(np.float32) / 255

    model = dist_pickle["model"]
    X_scaler = dist_pickle["X_scaler"]

    hot_windows = find_cars(img, Y_START_STOP[0], Y_START_STOP[1], SCALE, model,
                            X_scaler, ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, SPATIAL_SIZE, HIST_BINS, COLOR_SPACE, SPATIAL_FEAT, HIST_FEAT, HOG_FEAT)

    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    heat = add_heat(heat, hot_windows)
    heat = apply_threshold(heat, 1)

    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(draw_image, labels)

    return draw_img


def read_video(filename='project_video.mp4', saved=False):
    cap = cv2.VideoCapture(filename)
    # read dist pickle
    dist_pickle = pickle.load(open("model_linear.p", 'rb'))

    if saved:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
        output = "output-" + filename
        out = cv2.VideoWriter(output, fourcc, 20.0, (1280, 720))

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if frame is None:
            break

        img = detect(frame, dist_pickle)

        if saved:
            out.write(img)

        # Our operations on the frame come here
        # Display the resulting frame
        cv2.imshow('img', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def read_test_images():
    car = "data/vehicles/GTI_Far/image0004.png"
    notcar = "data/non-vehicles/GTI/image776.png"
    detect_car = "test_images/test1.jpg"
    sliding_windows = ["test_images/test4.jpg",
                       "test_images/test1.jpg", "test_images/test5.jpg"]
    dist_pickle = pickle.load(open("model_linear.p", 'rb'))

    def save(img, name, fname):
        filepath = "output_images/" + name + "-" + str(fname.split('/')[-1])
        cv2.imwrite(filepath, img)

    # car hog features
    car_img = cv2.imread(car)
    for ch in range(0, 3):

        features, hog_car = get_hog_features(car_img[:, :, ch], orient=ORIENT, pix_per_cell=PIX_PER_CELL,
                                             cell_per_block=CELL_PER_BLOCK, vis=True)
        save(hog_car * 255, 'hog_car_ch' + str(ch), car)
    # not car hog features
    notcar_img = cv2.imread(notcar)
    for ch in range(0, 3):
        features, hog_notcar = get_hog_features(notcar_img[:, :, ch], orient=ORIENT, pix_per_cell=PIX_PER_CELL,
                                                cell_per_block=CELL_PER_BLOCK, vis=True)
        save(hog_notcar * 255, 'hog_notcar_ch' + str(ch), notcar)
    # sliding windows
    for img_path in sliding_windows:
        img = cv2.imread(img_path)
        img = img.astype(np.float32) / 255

        draw_image = np.copy(img)
        model = dist_pickle["model"]
        X_scaler = dist_pickle["X_scaler"]
        hot_windows = find_cars(img, Y_START_STOP[0], Y_START_STOP[1], SCALE, model,
                                X_scaler, ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, SPATIAL_SIZE, HIST_BINS, COLOR_SPACE, SPATIAL_FEAT, HIST_FEAT, HOG_FEAT)

        sliding_window_img = draw_boxes(draw_image * 255, hot_windows)
        save(sliding_window_img, 'sliding_window', img_path)
        # heat map
        heat = np.zeros_like(img[:, :, 0]).astype(np.float)
        heat = add_heat(heat, hot_windows)
        heat = apply_threshold(heat, 1)
        heatmap = np.clip(heat, 0, 255)
        labels = label(heatmap)
        heat_map_img = draw_labeled_bboxes(draw_image, labels)
        save(heat * 255, 'heat_map', img_path)

    # detect a car
    detect_car_img = cv2.imread(detect_car)
    detect_car_img = detect(detect_car_img, dist_pickle)
    save(detect_car_img, 'detect_car', detect_car)


def main(args):
    #read_video(filename=args.fileinput, saved=args.save_video)
    read_test_images()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--fileinput',
                        type=str, help='finename of a video file')
    parser.add_argument('-s', '--save_video', action="store_true", default=False,
                        help='Either to save the output result or not')

    main(parser.parse_args())
