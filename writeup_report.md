## Writeup Template

---

**Vehicle Detection Project**

The goals of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and binned color features, as well as histograms of color, to your HOG feature vector
* Normalize the features and randomize a selection for training and testing
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/hog_feature.jpg
[image2]: ./output_images/binned_feature.jpg
[image3]: ./output_images/color_histogram.jpg
[image4]: ./output_images/normalization.jpg
[image5]: ./output_images/sliding_window1.jpg
[image6]: ./output_images/sliding_window2.jpg
[image7]: ./output_images/test1.jpg
[image8]: ./output_images/test3.jpg
[image9]: ./output_images/test4.jpg
[image10]: ./output_images/heatmap.jpg


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/513/view)  individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown.    

You're reading it!

### Feature Extraction

#### 1. Explain how you extracted HOG features from the training images.

The code for this step is contained in the `get_hog_features()` function of the IPython notebook.  

I extracted the HOG features from one of each of the `vehicle` and `non-vehicle` classes. Here is an example with the HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`

![alt text][image1]

I tested several parameters. For the video processing, I converted the raw image into YCrCb and extracted the HOG features for all the channels. In the end, I chose `orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`.

#### 2. Explain how you extracted the spatially binned feature and color histogram

The codes for this step are contained in the `color_hist()` and `bin_spatial()` functions. 

I converted the RGB raw image into YCrCb and resized it from 64x64 to 32x32. Then I converted it into a one-dimensional feature vector. Here is an example for the spatial binning.

![alt text][image2]

Then I concatenated the color channels into one feature vector. Here is an example with 128 bins in the range from 0 to 255.

![alt text][image3]


#### 3. Describe how you trained a classifier using your selected features.

I combined the HOG, color histogram and spatial binned features into one feature vector. The length of the feature vector is 4356. Then I normalized this vector. Here is an example for the raw feature vector and the normalized vector.

![alt text][image4]

For the classifier training, I read all the vehicle images from the KITTI data set and the non-vehicle images from the GTI data set. Then I extracted the combined features from these images. It took 61.24 seconds to extract these features.

I separated the data set into training and validation set. Then, I used the linear support vector classifier in the `scikit-learn`to classify the vehicle and non-vehicle images. The accuracy of the trained classifier is 0.99.


### Sliding Window Search

#### 1. Describe how you implemented a sliding window search.  

I defined the `slide_window` function for the sliding window search. I decided to search the window positions at different scales. 

Here is an example with a window size 64x64 and 50% overlap. 

![alt text][image5]


Here is an example with a window size 192x192 and 50% overlap. 

![alt text][image6]


#### 2. Show some examples of test images to demonstrate how your pipeline is working. 

I used the sliding window search with four window sizes: 48x48, 64x64, 96x96, 128x128. Here are some test results.
![alt text][image7]

![alt text][image8]

![alt text][image9]

#### 3. HOG Sub-sampling Window Search

The sliding window search needs to calculate the HOG features within each window in the same image. A more efficient approach is to extract the HOG features once and these features can be sub-sampled for all the overlapping windows. The codes for this new method are listed in the `find_cars` function.

#### 4. Apply Heatmap 

I built a heatmap from the sliding window detection results to reduce the false positives.To make a heat-map, I added "heat" (+=1) for all pixels within windows where a positive detection is reported by the linear support vector classifier. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a single image.

![alt text][image10]



### Video Implementation

#### 1. Provide a link to your final video output. 
Here's a [link](./project_video_out.mp4) to my video result. The video output was generated with detected vehicle positions drawn in bounding boxes.


#### 2. Describe how you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video in the `detection_result` class. I stored detection results from 15 consecutive frames. I added "heat" (+=1) for all pixels within windows where a positive detection is reported by the linear support vector classifier. All the detected pixels with a "heat" value less than a threshold will be filtered out.

---

### Discussion

#### 1. Briefly discuss any problems you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In this project, I found out that the parameter tuning for the feature extraction process is very time-consuming. A small changes in the parameters, such as the window size and the overlap, may cause huge difference in the result. Additionally, it took about 20 minutes to process a 50-second video, which means this method cannot be used for real-time vehicle detection.

This pipeline will fail in the following conditions:

* The ambient light is too dark or too strong
* The shape of the vehicle is not recorded in the training data set
* Two vehicle figures overlap each other on the image

The possible improvement methods are:

* Use neural network method or the non-linear support vector machine to train the classifier
* Use a larger training data set through data augmentation
* Calculate the centroid and the predict the vehicle location and speed
* The sliding window search in a new frame can start from the detected window results in the previous frame
* Use a decision tree to determine the relative importance of features and reduce the features with less contribution
