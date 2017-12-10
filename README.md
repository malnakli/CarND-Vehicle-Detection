## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car_not_car]: ./examples/car_not_car.png
[hog_car1]: ./output_images/hog_car_ch0.png
[hog_car2]: ./output_images/hog_car_ch1.png
[hog_car3]: ./output_images/hog_car_ch2.png
[hog_notcar1]: ./output_images/hog_notcar_ch0.png
[hog_notcar2]: ./output_images/hog_notcar_ch1.png
[hog_notcar3]: ./output_images/hog_notcar_ch2.png

[sliding_window1]: ./output_images/sliding_window-test1.jpg
[sliding_window2]: ./output_images/sliding_window-test4.jpg
[sliding_window3]: ./output_images/sliding_window-test5.jpg

[heat1]: ./output_images/frame-1-heat.png
[heat2]: ./output_images/frame-2-heat.png
[heat3]: ./output_images/frame-3-heat.png
[heat4]: ./output_images/frame-4-heat.png
[heat5]: ./output_images/frame-5-heat.png
[heat6]: ./output_images/frame-6-heat.png
[fram1]: ./output_images/frame-1.png
[fram2]: ./output_images/frame-2.png
[fram3]: ./output_images/frame-3.png
[fram4]: ./output_images/frame-4.png
[fram5]: ./output_images/frame-5.png
[fram6]: ./output_images/frame-6.png

[heat_output] ./output_images/frame-7-heat-output.png
[detect]: ./output_images/frame-7-output.jpg

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for extracted HOG features is contained in lines # 21 through # 39 in `utils.py`. 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][car_not_car]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][hog_car1] ![alt text][hog_notcar1]
![alt text][hog_car2] ![alt text][hog_notcar2]
![alt text][hog_car3] ![alt text][hog_notcar3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters such as I decrease pixels_per_cell = (2,2) and I selected randoms orientations between 6 and 12, however, I found that the best combinations that balance between accuracy and speed is the one I mentioned earlier. For example, select lower number for pixels_per_cell it could increase the accuracy a bit, but it takes longer to extract HOG features.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

All the code need to train a model in `train.py` file.
1. I loaded car and non-cars data.
2. Extract features (HOG, color and spatial features) for both data by using `extract_features` function in `utils.py` in lines # 66 through # 106
3. split data into training and testing manual because some of the vehicles data are the same, which appear more than once, but typically under significantly different lighting/angle from other instances.
4. I trained a SVM using `sklearn.svm.SVC` with the following parameters 
# TODO fill the parameters
`kernel=linear`,`C=5`,`gamma=auto`

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented by sliding window by using sub-sampling, the entire code can be found in line # 123 through # 214 in `utils.py`. 

1. Crop unnecessary region of the image from top and bottom (400px, 656px)
2. convert the image if the training images have been converted
3. resize the image by scale it down (33.3%)
4. get the three channels of `YCrCb`
5. Compute individual channel HOG features for the entire image
6. loop through the entire image and of each cell block do the following:
    1. Obtain the hog features if it turn on, by combining HOG features for each individual channel 
    2. Obtain spatial features if turn on
    3. Obtain color features if turn on
    4. concatenate features and apply normalization since it was applied on training data.
    5. predict if a car or not, 
    6. if the cell block was a predicted to be a car then add its box coordinates to an array.
9. return boxes that predicted to be a car

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][sliding_window1] ![alt text][sliding_window2] ![alt text][sliding_window3]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output-project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:



### Here are six frames and their corresponding heatmaps:
![alt text][heat1] ![alt text][frame1]
![alt text][heat2] ![alt text][frame2]
![alt text][heat3] ![alt text][frame3]
![alt text][heat4] ![alt text][frame4]
![alt text][heat5] ![alt text][frame5]
![alt text][heat6] ![alt text][frame6]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][heat_output]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][detect]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  


Tracker:
-  Accept frame and boxes (position of cars in that frame).
- 