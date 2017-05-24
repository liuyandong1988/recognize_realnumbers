cmake_minimum_required(VERSION 2.8.3)
project(image)

## Find catkin macros and libraries

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  rospy
  std_msgs
  image_transport
)

## Declare a catkin package  
catkin_package()
    
## Build imageCapRec 
include_directories(include ${catkin_INCLUDE_DIRS})  

## Declare a C++ executable
# add_executable(image_node src/image_node.cpp)
 add_executable(imageCapture src/imageCapture.cpp)
 target_link_libraries(imageCapture
   ${catkin_LIBRARIES}
 )

 add_executable(imageCapRec src/imageCapRec.cpp)
 target_link_libraries(imageCapRec
   ${catkin_LIBRARIES}
 )

 add_executable(imageCapreal src/imageCapreal.cpp)
 target_link_libraries(imageCapreal
   ${catkin_LIBRARIES}
 )

