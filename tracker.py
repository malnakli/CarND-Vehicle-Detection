import numpy as np
import cv2
from utils import *
from scipy.ndimage.measurements import label


class Track(object):

    frame_no = 0
    cars = []
    disply_cars = []

    @classmethod
    def next_frame(cls, frame, bboxes):
        cls.frame_no += 1
        [cls.cars.append(Car(bbox, cls.frame_no)) for bbox in bboxes]
        if cls.frame_no % 10 == 0:
            cls.filter_cars(frame.shape)

        for car in cls.disply_cars:
            if car.display:
                cv2.rectangle(frame, car.bbox[0], car.bbox[1], (0, 0, 255), 6)

        return frame

    @classmethod
    def filter_cars(cls, img_shape):
        hot_windows = [car.bbox for car in cls.cars]
        heat = np.zeros(img_shape[:-1]).astype(np.float)
        heat = add_heat(heat, hot_windows)
        heat = apply_threshold(heat, 16)
        heatmap = np.clip(heat, 0, 255)
        labels = label(heatmap)
        cls.cars = []
        cls.bbox_from_labels(labels)
        [cls.update_display_cars(car) for car in cls.cars]

        def keep_display_car(car):
            if car.display:
                if np.absolute(cls.frame_no - car.last_frame_seen) < 61:
                    return True
            elif np.absolute(cls.frame_no - car.last_frame_seen) < 31:
                return True

            return False

        cls.disply_cars = [
            car for car in cls.disply_cars if keep_display_car(car)]

    @classmethod
    def update_display_cars(cls, new_car):
        if len(cls.disply_cars) == 0:
            cls.disply_cars.append(new_car)
        else:
            car_exist = False
            for car in cls.disply_cars:
                if car.is_same_car(new_car):
                    car.num_of_seen += 1
                    car.last_frame_seen = cls.frame_no
                    car_exist = True
                    if car.num_of_seen > 3:
                        car.display = True
                    break
            if not car_exist:
                cls.disply_cars.append(new_car)

    @classmethod
    def bbox_from_labels(cls, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            bbox = bbox_from_label(labels[0], car_number)
            # cls.check_car(bbox)
            new_car = Car(bbox, cls.frame_no)
            if new_car.is_it_a_car():
                cls.cars.append(new_car)


class Car(object):

    def __init__(self, bbox, frame_no):
        self.bbox = bbox
        self.num_of_seen = 1
        self.start_x = self.bbox[0][0]
        self.stop_x = self.bbox[1][0]
        self.start_y = self.bbox[0][1]
        self.stop_y = self.bbox[1][1]
        self.last_frame_seen = frame_no
        self.first_frame_seen = frame_no
        self.display = False
        self.trackable = False  # when True then start track

    def is_same_car(self, car):
        if self.trackable:
            return False

        labels = self._label_overlap(car)

        if labels[1] == 1:
            # adjust the box of the car if it only seen less than 6, otherwise trackable

            if self.start_x > car.start_x:
                if self.stop_x > car.stop_x:
                    self.adjust_box(car, mode="bigger_or_smaller")
                else:
                    self.adjust_box(car, mode="righter")
            else:
                if self.stop_x > car.stop_x:
                    self.adjust_box(car, mode="lefter")
                else:
                    self.adjust_box(car, mode="bigger_or_smaller")

            return True

        return False

    def is_it_a_car(self):
        # the box is very small
        if self.stop_y - self.start_y < 20:
            return False
        elif self.stop_x - self.start_x < 20:
            return False
        return True

    def _label_overlap(self, car):
        """
        check if the boxes are inside each other
        """
        heat = np.zeros((720, 1280)).astype(np.float)
        heat[self.bbox[0][1]:self.bbox[1][1],
             self.bbox[0][0]: self.bbox[1][0]] = 1
        heat[car.bbox[0][1]: car.bbox[1][1],
             car.bbox[0][0]: car.bbox[1][0]] += 1

        heat = apply_threshold(heat, 1)
        heatmap = np.clip(heat, 0, 255)

        labels = label(heatmap)
        return labels

    def adjust_box(self, car, mode="None"):
        new_start_x = np.mean([car.start_x, self.start_x]).astype(np.int)
        if mode == 'lefter':
            # by how much start_x position has moved then apply to stop_x
            self.stop_x -= np.absolute(new_start_x - self.start_x)
            self.start_x = new_start_x
        elif mode == 'righter':
            # by how much start_x position has moved then apply to stop_x
            self.stop_x += np.absolute(new_start_x - self.start_x)
            self.start_x = new_start_x
        elif mode == 'bigger_or_smaller':
            new_start_y = np.mean([car.start_y, self.start_y]).astype(np.int)
            new_stop_x = np.mean([car.stop_x, self.stop_x]).astype(np.int)
            new_stop_y = np.mean([car.stop_y, self.stop_y]).astype(np.int)
            self.start_x = new_start_x
            self.start_y = new_start_y
            self.stop_x = new_stop_x
            self.stop_y = new_stop_y

        self.bbox = ((self.start_x, self.start_y), (self.stop_x, self.stop_y))
