#ifndef LOCATOR_UTIL_H
#define LOCATOR_UTIL_H

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <time.h>

using cv::Mat;
using cv::Vec3b;
using cv::Rect;
using std::string;
using std::vector;

enum projectionID {
    VERTICAL, HORIZONTAL
};
struct Interval {
    int begin;
    int length;
};

struct Manuscript {
    string fileName;
    Mat image;
};

// Function Headers
int getIntensity(Vec3b pixel);

Vec3b getPixel(Mat &image, int x, int y);

void setPixel(Mat &image, int x, int y, Vec3b pixel);

void binaryNeighbourhoodMask(Mat &input, Mat &output, int minActivity);

vector<Interval>
intensityProjectionFilter(Mat &input, Mat &output, int projectionID, int xMin, int xMax, int yMin, int yMax,
                          int mapToZeroThreshold, int minLength);

vector<Mat> extractImages(Mat &image, std::vector<Rect> boundingRects);

void detectChineseLetters(Manuscript &manuscript);

double intensityFluctuations(Mat &input, int projectionID);

void processManuscripts(string file);

timespec diff(timespec start, timespec end);

#endif
