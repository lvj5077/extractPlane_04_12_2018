# 增加PCL库的依赖
FIND_PACKAGE( PCL REQUIRED )

list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4") # use this in Ubuntu 16.04
# 增加opencv的依赖

SET("OpenCV_DIR" "/home/jin/Packages/opencv-2.4.13/build")
FIND_PACKAGE( OpenCV REQUIRED )


ADD_DEFINITIONS( ${PCL_DEFINITIONS} )
INCLUDE_DIRECTORIES( ${PCL_INCLUDE_DIRS}  )
LINK_LIBRARIES( ${PCL_LIBRARY_DIRS} )

# add_executable (planar_segmentation planar_segmentation.cpp)
# target_link_libraries (planar_segmentation ${PCL_LIBRARIES})

ADD_EXECUTABLE( extractPlane extractPlane.cpp )
TARGET_LINK_LIBRARIES( extractPlane ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )