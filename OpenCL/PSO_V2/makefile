CC= g++
CFLAGS = -o
STD = -std=c++11
SOURCES = main.cpp
EXECUTABLE = main

FRAMEWORK = -framework OpenCL

#LIBPATH = -L/usr/local/lib
#LIBPATH += -L/usr/local/Cellar/armadill/5.200.1/lib
#LIB = -lopencv_core -lopencv_highgui -lopencv_imgproc

INCLUDEPATH = -I/usr/local/Cellar/eigen/3.2.6/include/eigen3
INCLUDEPATH += -I ./Cpp_common/
#INCLUDEPATH += -I/usr/local/include
#INCLUDEPATH += -I/usr/local/Cellar/armadillo/5.200.1/include

all: 
	$(CC) $(STD) $(CFLAGS) $(EXECUTABLE) $(SOURCES) $(INCLUDEPATH) $(FRAMEWORK)
	./$(EXECUTABLE)

