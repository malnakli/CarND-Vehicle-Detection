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
           # cls.update_car(frame)
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
                        car.last_frame_seen = 0  # to be delete

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
        cls.bbox_from_labels(labels, True)

    @classmethod
    def bbox_from_labels(cls, labels, from_update_car=False):
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            bbox = bbox_from_label(labels[0],car_number)
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
      
        labels = self._label_overlap(car)

        if labels[1] == 1:
            self.bbox = bbox_from_label(labels[0],1)
            self.start = self.bbox[0][0]
            self.stop = self.bbox[1][0]
            return True
        elif self._close_boxes(car):
            self.start = np.mean([self.start,car.start]).astype(np.int)
            self.stop = np.mean([self.stop,car.stop]).astype(np.int)
            self.bbox = ((self.start,self.bbox[0][1]),(self.stop,self.bbox[1][1]))
            return True

        return False
    
    def _label_overlap(self,car):
        """
        check if the boxes are inside each other
        """
        heat = np.zeros((720,1280)).astype(np.float)
        heat[car.bbox[0][1]:car.bbox[1][1],
                 car.bbox[0][0]:car.bbox[1][0]] = 1
        heat[self.bbox[0][1]:self.bbox[1][1],
                 self.bbox[0][0]:self.bbox[1][0]] = 1
        
        cv2.imwrite("temp/heatmap.png", heat * 255)
        heatmap = np.clip(heat, 0, 255)
        
        labels = label(heatmap)
        return labels

    def _close_boxes(self,car):
        # it has to be 2 because there are self and other car to compare
        if np.absolute(self.stop - car.start) < 20:
            return True
        elif np.absolute(self.start - car.stop) < 20:
            return True

        return False