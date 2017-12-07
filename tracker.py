import numpy as np
import cv2
from utils import *
from scipy.ndimage.measurements import label


class Track(object):

    frame_no = 0
    cars = []

    @classmethod
    def next_frame(cls, frame, labels):
        cls.frame_no += 1
        cls.bbox_from_labels(labels)

        if cls.frame_no % 3 == 0:
            cls.update_car(frame)
            cls.cars = [x for x in cls.cars if not cls.delete_car(x)]
            for car in cls.cars:
                cv2.rectangle(frame, car.bbox[0], car.bbox[1], (0, 255, 0), 6)
        else:
            for car in cls.cars:
                cv2.rectangle(frame, car.bbox[0], car.bbox[1], (0, 0, 255), 6)

        return frame

    @classmethod
    def check_car(cls, bbox, from_update_car=False):
        new_car = Car(bbox, cls.frame_no)

        if len(cls.cars) == 0:
            cls.cars.append(new_car)
        else:
            car_exist = False
            if from_update_car:
                for car in cls.cars:
                    # car.last_frame_seen = 0  # to be delete
                    if new_car.is_same_car(car):
                        new_car.num_of_seen += car.num_of_seen

            else:
                for car in cls.cars:
                    if car.is_same_car(new_car):
                        car.num_of_seen += 1
                        car.last_frame_seen = cls.frame_no
                        car_exist = True
                        break
            if not car_exist:
                cls.cars.append(new_car)

    @classmethod
    def delete_car(cls, car):
        number_of_frame_shown = car.last_frame_seen - car.first_frame_seen
        total_frame = cls.frame_no - car.first_frame_seen
        last_time_car_seen = cls.frame_no - car.last_frame_seen
        # delete the car if it was not seen for 2 frame consecutive
        if (last_time_car_seen > 2 and number_of_frame_shown < 3):
            return True

        return False

    @classmethod
    def update_car(cls, frame):
        heat = np.zeros_like(frame[:, :, 0]).astype(np.float)
        for car in cls.cars:
            heat[car.bbox[0][1]:car.bbox[1][1],
                 car.bbox[0][0]:car.bbox[1][0]] = 1

        cv2.imwrite("temp/heat.png", heat * 255)
        heatmap = np.clip(heat, 0, 255)
        labels = label(heatmap)
        print(labels[1])
        cls.bbox_from_labels(labels, True)

    @classmethod
    def bbox_from_labels(cls, labels, from_update_car=False):
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                    (np.max(nonzerox), np.max(nonzeroy)))
            cls.check_car(bbox, from_update_car)


class Car(object):

    def __init__(self, bbox, frame_no):
        self.bbox = bbox
        self.num_of_seen = 1
        self.start = self.bbox[0][0]
        self.stop = self.bbox[1][0]
        self.last_frame_seen = frame_no
        self.first_frame_seen = frame_no

    def is_same_car(self, car):
        if self.start - 20 <= car.start <= self.start + 20 and self.stop - 20 <= car.stop <= self.stop + 20:
            # self.start = np.mean([self.start, car.start])
            # self.stop = np.mean([self.stop, car.stop])

            return True

        return False
