Show some opencv examples, all these examples are test based one below environment.

OS: Ubuntu 24.04 with g++ 13.3
RTX 4090 with CUDA driver 570 and toolkit 12.8
opencv 4.x built with cuda and codacodec
some dependencies libs 

![device image](stereo_camera.jpg)

1 stereo_distance:  in this example, a stereo camera is used to get the distance, and the result show the left picture and right picture and also the disparity picture and distance picture. In the distance picture some color are used to show far and near, near is read and far is blue, from near to far is red, yellow, green and blue.
![stereo image](stereo_distance/stereo_distance.png)