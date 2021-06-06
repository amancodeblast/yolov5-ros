after clonning do catkin_make in the perception module directory

## Build

Clone the repo and run

* rename the repo to perception_module
* cd perception_module
* catkin_make

## How to run the node
rosrun object_detection image_pub
## How to get the results
rostopic list
you will get the output as /image_converter/output_video
rviz rviz 
add the topic for this and choose /image_converter/output_video as image
