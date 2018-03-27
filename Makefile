GPP=g++ -std=c++11 -o3 -Wall -Wl,-rpath,/usr/local/lib/ -I/usr/local/include/  /usr/local/lib/libopencv_calib3d.so /usr/local/lib/libopencv_core.so  /usr/local/lib/libopencv_features2d.so  /usr/local/lib/libopencv_flann.so  /usr/local/lib/libopencv_highgui.so  /usr/local/lib/libopencv_imgcodecs.so /usr/local/lib/libopencv_imgproc.so /usr/local/lib/libopencv_xfeatures2d.so -o
All :	train.o test.o logorec.o
	$(GPP) train.bin train.o logorec.o
	$(GPP) test.bin test.o logorec.o
train.o:	trainWithLogos.cpp  libLogoRecog.h
	$(GPP) train.o -c trainWithLogos.cpp
test.o:	runtimeExample.cpp  libLogoRecog.h
	$(GPP) test.o -c runtimeExample.cpp
logorec.o:	libLogoRecog.h libLogoRecog.cpp
	$(GPP) logorec.o -c libLogoRecog.cpp
clean:
	rm -rf *.o *.bin

