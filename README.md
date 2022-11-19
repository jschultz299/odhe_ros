# Robot-Assisted Feeding - ODHE ROS Package

This is the ros package for the ODHE Grant Funded Project
Robot-Assisted Feeding (RAF) for Individuals with Spinal Cord Injury

This ros workspace has been used to develop the whole project,
so different iterations of the project require different program files.

This repo is not intended to be cloned into a ros workspace and be useful right away. In the future, I will upload a cleaned and maintained repo for this purpose.

## Head-Mounted System
The first iteration of the project used the iSCAN etl-600 wearable
eye tracker. We used a faster-RCNN object detection network to detect cups, 
bowls, plates, forks, and spoons. The network used the iSCAN's head-mounted 
camera as the input. To demo the system, we placed a cup in
the scene. The user then directs their gaze to the cup. When they are 
ready, the user issues a voice command such as "grab the cup". Baxter then 
picks up the cup that the user was looking at and delivers it to in front 
of the user's mouth. They then move their body to sip through a straw in 
the cup. 

<img src="https://github.com/jschultz299/odhe_ros/blob/main/images/IMG_7340.png" width=40%>

<img src="https://github.com/jschultz299/odhe_ros/blob/main/images/Object%20Detection%20Example.gif" width=50%>

I actually don't think this demonstration will work in this 
workspace because the launch file is in a different workspace. Originally, 
we were using Ubuntu 18.04 because we couldn't get Baxter to work with ROS 
Noetic and Python3. Then, I put in the effort to convert Baxter's code to 
Python3 and I switched to Ubuntu 20.04. The workspace did not get 
transferred over, but some of the files did. Below are some of the files 
associated with this project iteration.

### Scripts:
```bash
cup_demo.py
```
The main file for the cup demo. Handles all code logic, particularly the robot's motion.

```bash
cup_demo_short.py   
```
I think this version was supposed to only perform one reach, or some other shortened version for demonstration purposes. It looks very similar to the code above, however.

```bash
record_data.py
```
Records the iSCAN information and robot data.

```bash
serial_read.py  
```
Reads the serial data from the iSCAN.

## Tablet Interface System
The second iteration of this project got rid of the head-mounted eye 
tracker. Also, we decided to interact with different food items on a 
plate, instead of dishes and utensils. For the interface, we used a 
tablet monitor with a Tobii Eye tracker 4 mounted on the bottom. We 
detected objects using a mask-RCNN object detection network from 
detectron2. The network used an Intel L515 LIDAR depth camera as the 
input. The plane of the table was defined using an AprilTag fiducial 
marker. We used Talon to interface with the eyetracker. Talon also 
has the ability to allow the user to issue commands. The depth camera is 
mounted to Baxter's wrist. The camera videos food items on a plate, and 
the image is sent to the tablet with food item outlines drawn. The person 
uses their eye movements to direct a cursor to a food item. After a short 
dwell time, the food item is selected. Baxter picks up the food item and 
then turns to video the user's face. We used the face-alignment ROS 
package for facial keypoint detection. When the user open's their mouth, 
Baxter approaches the user's face and releases the food item. 

Check out a demo video [here](https://youtu.be/AmBzfEcXVCc)!

<img src="https://github.com/jschultz299/odhe_ros/blob/main/images/System%20Overview.png" width=50%>
<img src="https://github.com/jschultz299/odhe_ros/blob/main/images/Evaluation.png" width=50%>
<img src="https://github.com/jschultz299/odhe_ros/blob/main/images/Baxter%20Arm.png" width=50%>
<img src="https://github.com/jschultz299/odhe_ros/blob/main/images/GUI.png" width=50%>
<img src="https://github.com/jschultz299/odhe_ros/blob/main/images/Object%20Detection.png" width=40%>

Below are the files associated with this project iteration.

### Launch File:
```bash
raf_study.launch
    - camera_multiple.launch
    - tag_detection.py
    - arm_camera_network.py
    - dlt.py
```

```bash
raf.launch 
```
Top-level launch file for the project.

```bash
camera_multiple.launch     
```
Launches both the LIDAR and STEREO depth cameras. Requires serial numbers.

```bash
tag_detection.py         
```
Detects the AprilTag Fiducial markers and draws the tag coordinate frame.

```bash
arm_camera_network_run.py   
```
Detects food items on the plate. 

```bash
dlt.py    
```
Defines the plane of the table and publishes DLT parameters.

### Scripts:
```bash
calibrate_camera.py     
```
Runs the logic for performing the calibration which defines the camera in robot coordinates.

```bash
raf.py            
```
Main project script file. Handles all experiment logic and robot motion.

```bash
realtime_ros.py  
```
Currently not in the project workspace. Located in face-alignment. This code detects the person's face.

### Other Files:
```bash
raf_grasp_demo.py  
```
Demonstrates picking up food items anywhere on the plate.

```bash
raf_setPos_demo.py  
```
Demonstrates picking up a food item in a set position and orientation.

```bash
raf_visualization.py  
```
Test code that draws GUI info for food item detection. This got integrated into raf.py now.

```bash
raf_visualize_grasp.py   
```
Nice visualization for the position and orientation of food item.
