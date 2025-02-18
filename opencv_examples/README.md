Show some opencv examples, all these examples are test based one below environment.

OS: Ubuntu 24.04 with g++ 13.3
RTX 4090 with CUDA driver 570 and toolkit 12.8, CudaDNN installed.
opencv 4.x built with cuda and codacodec ON
All dependencies libs installed

Stereo camera device is in the below picture:
![device image](stereo_camera.jpg)

1 stereo_distance:  in this example, a stereo camera is used to get the distance, and the result show the left picture and right picture and also the disparity picture and distance picture. In the distance picture some color are used to show far and near, near is read and far is blue, from near to far is red, yellow, green and blue.
![stereo image](stereo_distance/stereo_distance.png)

2 object_track: in this example, a stereo camera is used, a bottle of water is inside both camera's sight, and then choose the bottle of water as the target to track(you need selete the rect area of the bottle manually), the use openCV track algorithm CSRT to track it.

![object track jif](object_track/object_tracking.gif)