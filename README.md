# CarND Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


### Files Submitted & Code Quality

1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](model.py)  containing the script to create and train the model
* [drive.py](drive.py) for driving the car in autonomous mode
* [model.h5](model-004.h5) containing a trained convolution neural network (this model.h5 file is model-004.h5(which is one of many model-xxx.h5 files produced by tweaking the model parameters and running ```python model.py```))
* [output text file 1 while running ```python model.py``` on aws EC2 GPU instance](output-text-file1)
* [output text file 2 while running ```python model.py```](output-text-file2)
* [output text file 3 while running ```python model.py```](output-text-file3)
* [video.py](video.py) for converting the image files to video
* [README.md](README.md) summarizing the results

2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```python drive.py model.h5```

3. Submission code is usable and readable

The [model.py](model.py) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

[NVIDIA's End-to-End Deep Learning Model for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) is used. It consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 ([model.py](model.py) code lines 46-75) 

The data is normalized in the model using a Keras lambda layer ([model.py](model.py) code line 67)

The model includes ELU layers to introduce nonlinearity ([model.py](model.py) code lines 70-80)

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting ([model.py](model.py) code line 76). 

The model was trained and validated on different data sets to ensure that the model was not overfitting ([model.py](model.py) code line 142).

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer with the default learning rate (0.0001) ([model.py](model.py) code line 110 and line 148).

#### 4. Appropriate training data

I used [Udacity's SDC-ND Sample Training Data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) for Training [The Model](model.py)

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I started of with a simple CNN Model

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that there is less overfitting.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track or slammed the bridge and stopped or hit the tree and stopped. To improve the driving behavior in these cases, I tried to implement [NVIDIA's End-to-End Deep Learning Model for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)

After lots of trial and error , and tweaking the training set(from 80% to 90%) and test set (from 20% to 10%) and changing the batch size and learning rate.

#### 2. Final Model Architecture

Finally , [NVIDIA's End-to-End Deep Learning Model for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) was used. 

The Final Model Architecture consisted of convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 ([model.py](model.py) code lines 46-75) 

The data is normalized in the model using a Keras lambda layer ([model.py](model.py) code line 67)

The model includes ELU layers to introduce nonlinearity ([model.py](model.py) code lines 70-80)

The model contains dropout layers in order to reduce overfitting ([model.py](model.py) code line 76). 

The model was trained and validated on different data sets to ensure that the model was not overfitting ([model.py](model.py) code line 142).

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

| Layer (type) | Output Shape | Param #      | Connected to |
|--------------|--------------|--------------|--------------|
|lambda_1 (Lambda)| (None, 66, 200, 3) | 0 | lambda_input_1[0][0] |
|convolution2d_1 (Convolution2D)|  (None, 31, 98, 24) |   1824 |       lambda_1[0][0] |
|convolution2d_2 (Convolution2D) | (None, 14, 47, 36)  |  21636   |    convolution2d_1[0][0] |
|convolution2d_3 (Convolution2D) | (None, 5, 22, 48)  |   43248   |    convolution2d_2[0][0] |
|convolution2d_4 (Convolution2D) |(None, 3, 20, 64) |    27712    |   convolution2d_3[0][0] |
|convolution2d_5 (Convolution2D) | (None, 1, 18, 64)  |   36928   |    convolution2d_4[0][0] |
|dropout_1 (Dropout)    |          (None, 1, 18, 64)    | 0      |     convolution2d_5[0][0] |
|flatten_1 (Flatten)     |         (None, 1152)    |      0       |    dropout_1[0][0] |
|dense_1 (Dense)   |               (None, 100)     |      115300    |  flatten_1[0][0] |
|dense_2 (Dense)    |              (None, 50)     |       5050     |   dense_1[0][0] |
|dense_3 (Dense)    |              (None, 10)      |      510      |   dense_2[0][0] |
|dense_4 (Dense)     |             (None, 1)       |      11        |  dense_3[0][0] |

#### 3. Creation of the Training Set & Training Process

#### 4. Final Video

for recording or saving the images for video in folder [rn1](rn1)

```python drive.py model-004.h5 rn1```



for taking the recorded or saved images and making the video rn1.mp4 at 60 frames per second (default)

```python video.py rn1 ```  outputs [video(youtube)](https://youtu.be/gvwRCXzHGGs) / [video in the repo](Video-60fps.mp4)




for taking the recorded or saved images and making the video rn1.mp4 at 40 frames per second

```python video.py rn1 --fps 40``` outputs [video(youtube)](https://youtu.be/lEZAF99rWQI) / [video in the repo](Video-40fps.mp4)

