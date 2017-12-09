import numpy as np
import cv2
from utils import *
from scipy.ndimage.measurements import label


class Track(object):

    frame_no = 0
    cars = []
    disply_cars = []
    @classmethod
    def next_frame(cls, frame, labels):
        cls.frame_no += 1
        #cls.bbox_from_labels(labels)
        [cls.cars.append(Car(bbox,cls.frame_no)) for bbox in labels]
        if cls.frame_no % 5 == 0:
            cls.filter_cars()
            cls.disply_cars = [car for car in cls.disply_cars if car.car_disply > 0.1]
        
        for car in cls.disply_cars:
            if car.trackable:
                cv2.rectangle(frame, car.bbox[0], car.bbox[1], (0, 255, 0), 6)
            else:
                cv2.rectangle(frame, car.bbox[0], car.bbox[1], (0, 0, 255), 6)

        return frame

    @classmethod
    def filter_cars(cls):
        hot_windows = [car.bbox for car in cls.cars]
        heat = np.zeros((720,1280)).astype(np.float)
        heat = add_heat(heat, hot_windows)
        heat = apply_threshold(heat, 4)
        heatmap = np.clip(heat, 0, 255)
        labels = label(heatmap)
        cls.cars = []
        cls.bbox_from_labels(labels)
        #cls.disply_cars = [car for car in cls.cars if car.is_it_a_car()]
        [cls.update_disply_cars(car) for car in cls.cars]
        cls.calculate_car_disply_precentage()
    @classmethod
    def update_disply_cars(cls,new_car):
        if len(cls.disply_cars) == 0:
            cls.disply_cars = cls.cars
        else:
            car_exist = False
            for car in cls.disply_cars:
                    if car.is_same_car(new_car):
                        car.num_of_seen += 1
                        car.last_frame_seen = cls.frame_no
                        car.points = cls.calculate_car_points(car)
                        car_exist = True
                        break
            if not car_exist:
                cls.disply_cars.append(new_car)
                
    @classmethod
    def calculate_car_points(cls, car):

        num_of_frame_seen = car.num_of_seen 
        number_of_frame_shown = np.absolute(car.last_frame_seen - car.first_frame_seen) * .5
        last_frame_seen  = np.absolute(cls.frame_no - car.last_frame_seen)
        
        return num_of_frame_seen + number_of_frame_shown - last_frame_seen

    @classmethod
    def calculate_car_disply_precentage(cls):

        cars_points = [car.points for car in cls.disply_cars]
        # Compute softmax values for each sets of scores in cars_points
        cars_disply_percentages = np.exp(cars_points) / np.sum(np.exp(cars_points), axis=0)
        for index, car in enumerate(cls.disply_cars):
            car.car_disply = cars_disply_percentages[index]
            if (cars_disply_percentages[index] > .75):
                if car.is_it_a_car():
                    car.trackable = True
                else:
                    car.num_of_seen = 0

            print(cars_disply_percentages[index] * 100)
        

    @classmethod
    def bbox_from_labels(cls, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            bbox = bbox_from_label(labels[0],car_number)
            #cls.check_car(bbox)
            new_car = Car(bbox, cls.frame_no)
            if new_car.is_it_a_car():
                cls.cars.append(new_car)


class Car(object):

    def __init__(self, bbox, frame_no):
        self.bbox = bbox
        self.num_of_seen = 1
        self.start = self.bbox[0][0]
        self.stop = self.bbox[1][0]
        self.starty = self.bbox[0][1]
        self.stopy = self.bbox[1][1]
        self.last_frame_seen = frame_no
        self.first_frame_seen = frame_no
        self.points = 0 # number, which will be used to compare with other frames
        self.car_disply = 0.0 # change to disply car in percentage
        self.trackable = False # when True then start track

    def is_same_car(self, car):
      
        labels = self._label_overlap(car)

        if labels[1] == 1:
            self.bbox = bbox_from_label(labels[0],1)
            self.start = self.bbox[0][0]
            self.stop = self.bbox[1][0]
            return True
        # elif self._close_boxes(car):
        #     self.bbox = ((self.start,self.bbox[0][1]),(self.stop,self.bbox[1][1]))
        #     return True

        return False
    def is_it_a_car(self):
        # the box is very big
        if self.stopy - self.starty < 20:
            return False
        elif self.stop - self.start < 20:
            return False
        return True

    def _label_overlap(self,car):
        """
        check if the boxes are inside each other
        """
        heat = np.zeros((720,1280)).astype(np.float)
        heat[car.bbox[0][1]:car.bbox[1][1],
                 car.bbox[0][0]:car.bbox[1][0]] = 1
        heat[self.bbox[0][1]:self.bbox[1][1],
                 self.bbox[0][0]:self.bbox[1][0]] += 1
        
        heat = apply_threshold(heat, 2)
        heatmap = np.clip(heat, 0, 255)
        
        labels = label(heatmap)
        return labels