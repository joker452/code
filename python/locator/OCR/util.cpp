#include "util.h"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

timespec diff(timespec start, timespec end) {
    timespec dif;
    if ((end.tv_nsec - start.tv_nsec) < 0) {
        dif.tv_sec = end.tv_sec - start.tv_sec - 1;
        dif.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    } else {
        dif.tv_sec = end.tv_sec - start.tv_sec;
        dif.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    return dif;
}

void processManuscripts(string file) {
    Manuscript manuscript;
    manuscript.fileName = file;
    manuscript.image = imread(manuscript.fileName);
    detectChineseLetters(manuscript);

}

void detectChineseLetters(Manuscript &manuscript) {
    cout << "Detecting letters in " << manuscript.fileName << endl;
    size_t found = manuscript.fileName.find_last_of("/\\");
    if (found == string::npos) {
        cerr << "path parser fail!" << endl;
        exit(-1);
    }
    string file_part_name = manuscript.fileName.substr(found + 1);
    found = file_part_name.find_last_of('.');
    if (found == string::npos) {
        cerr << "file name parse fail!" << endl;
        exit(-1);
    }
    file_part_name = file_part_name.substr(0, found);
    Mat processingImage = manuscript.image.clone();
    Mat intensityThresholdImage, intensityProjectionImage;
    Mat detectionImage = manuscript.image.clone();

    binaryNeighbourhoodMask(processingImage, intensityThresholdImage, 2);


    //Perform Intensity Projection to seperate vertically and horizontally
    int mapToZeroThreshold = 15;
    int minLength = 15;
    vector<Rect> boundingRects;
    vector<Interval> verticalRegions = intensityProjectionFilter(intensityThresholdImage, intensityProjectionImage,
                                                                 VERTICAL, 0, intensityThresholdImage.cols, 0,
                                                                 intensityThresholdImage.rows, mapToZeroThreshold,
                                                                 minLength);
    for (int i = 0; i < verticalRegions.size(); i++) {
        vector<Interval> horizontalRegions = intensityProjectionFilter(intensityProjectionImage,
                                                                       intensityProjectionImage, HORIZONTAL,
                                                                       verticalRegions.at(i).begin,
                                                                       verticalRegions.at(i).begin +
                                                                       verticalRegions.at(i).length,
                                                                       0, intensityThresholdImage.rows,
                                                                       mapToZeroThreshold, minLength);
        for (int j = 0; j < horizontalRegions.size(); j++) {
            double expandFactor = 0.3;
            double x = verticalRegions.at(i).begin;
            double y = horizontalRegions.at(j).begin;
            double width = verticalRegions.at(i).length;
            double height = horizontalRegions.at(j).length;

            double expandX = width * expandFactor;
            double expandY = height * expandFactor;
            width += expandY;
            height += expandX;

            x -= 0.5 * expandY;
            if (x < 0)
                x = 0;
            if (x > manuscript.image.cols)
                x = manuscript.image.cols - 1 - width;
            y -= 0.5 * expandX;
            if (y < 0)
                y = 0;
            if (y > manuscript.image.rows)
                y = manuscript.image.rows - 1 - height;
            if (x + width > manuscript.image.cols)
                width = manuscript.image.cols - 1 - x;
            if (y + height > manuscript.image.rows)
                height = manuscript.image.rows - 1 - y;
            boundingRects.push_back(Rect(x, y, width, height));
        }

    }

    //Extract Letter Images (Filter false positives with low fluctuations (borders) to obtain a better training set)
    vector<Mat> chineseLetterImages;
    chineseLetterImages = extractImages(intensityThresholdImage, boundingRects);
    for (int i = 0; i < chineseLetterImages.size(); i++) {
        vector<Point> nonZeroPixels;
        for (int x = 0; x < chineseLetterImages.at(i).cols; x++) {
            for (int y = 0; y < chineseLetterImages.at(i).rows; y++) {
                if (getIntensity(getPixel(chineseLetterImages.at(i), x, y)) != 0) {
                    nonZeroPixels.push_back(Point(boundingRects.at(i).x,
                                                  boundingRects.at(i).y) + Point(x, y));
                }
            }
        }
        boundingRects.at(i) = boundingRect(nonZeroPixels);

        if (intensityFluctuations(chineseLetterImages.at(i), VERTICAL) <= 1 ||
            intensityFluctuations(chineseLetterImages.at(i), HORIZONTAL) <= 1) {
            chineseLetterImages.erase(chineseLetterImages.begin() + i);
            boundingRects.erase(boundingRects.begin() + i);
            i--;
        }
    }

    // post-processing
    for (unsigned long i = 0; i < boundingRects.size(); ++i) {
        int w = boundingRects[i].width;
        int h = boundingRects[i].height;
        if (w < 20 || w > 200 || h < 20 || h > 250) {
            boundingRects.erase(boundingRects.begin() + i);
            i--;
        }
    }
    chineseLetterImages = extractImages(manuscript.image, boundingRects);
    // save detection image
    Scalar color = Scalar(255, 255, 255);

    for (int i = 0; i < chineseLetterImages.size(); i++) {
        rectangle(detectionImage, boundingRects.at(i), color, 2);
    }
    imwrite("./res_img/" + file_part_name + ".jpg", detectionImage);

    // write results to file
    chineseLetterImages = extractImages(manuscript.image, boundingRects);
    ofstream out(("./res_txt/" + file_part_name + ".txt").c_str());
    for (int i = 0; i < chineseLetterImages.size(); ++i)
        out << boundingRects.at(i).x << " " << boundingRects.at(i).y
            << " " << boundingRects.at(i).x + boundingRects.at(i).width
            << " " << boundingRects.at(i).y + boundingRects.at(i).height << endl;
}

Vec3b getPixel(Mat &image, int x, int y) {
    Vec3b *pixel = &image.at<Vec3b>(Point(x, y));
    return Vec3b(pixel->val[2], pixel->val[1], pixel->val[0]);
}

void setPixel(Mat &image, int x, int y, Vec3b pixel) {
    image.at<Vec3b>(Point(x, y)) = Vec3b(pixel.val[2], pixel.val[1], pixel.val[0]);
}

int getIntensity(Vec3b pixel) {
    return int((pixel.val[0] + pixel.val[1] + pixel.val[2]) / 3.0);
}

void binaryNeighbourhoodMask(Mat &input, Mat &output, int minActivity) {
    output = input.clone();
    for (int x = 0; x < input.cols; x++) {
        for (int y = 0; y < input.rows; y++) {
            int activeNeighbors = 0;

            if (x - 1 >= 0) {
                if (getIntensity(getPixel(input, x - 1, y)) > 0) {
                    activeNeighbors++;
                }
            }
            if (x + 1 < input.cols) {
                if (getIntensity(getPixel(input, x + 1, y)) > 0) {
                    activeNeighbors++;
                }
            }
            if (y - 1 >= 0) {
                if (getIntensity(getPixel(input, x, y - 1)) > 0) {
                    activeNeighbors++;
                }
            }
            if (y + 1 < input.rows) {
                if (getIntensity(getPixel(input, x, y + 1)) > 0) {
                    activeNeighbors++;
                }
            }
            if (x - 1 >= 0 && y - 1 >= 0) {
                if (getIntensity(getPixel(input, x - 1, y - 1)) > 0) {
                    activeNeighbors++;
                }
            }
            if (x + 1 < input.cols && y + 1 < input.rows) {
                if (getIntensity(getPixel(input, x + 1, y + 1)) > 0) {
                    activeNeighbors++;
                }
            }
            if (x - 1 >= 0 && y + 1 < input.rows) {
                if (getIntensity(getPixel(input, x - 1, y + 1)) > 0) {
                    activeNeighbors++;
                }
            }
            if (x + 1 < input.cols && y - 1 >= 0) {
                if (getIntensity(getPixel(input, x + 1, y - 1)) > 0) {
                    activeNeighbors++;
                }
            }

            if (activeNeighbors < minActivity) {
                setPixel(output, x, y, Vec3b(0, 0, 0));
            }
        }
    }
}


vector<Interval>
intensityProjectionFilter(Mat &input, Mat &output, int projectionID, int xMin, int xMax, int yMin, int yMax,
                          int mapToZeroThreshold, int minLength) {
    output = input.clone();

    vector<int> intensityProjection;
    vector<Interval> intervals;

    if (projectionID == VERTICAL) {
        //Compute average intensity at each row or column and erase if value below mapToZero threshold
        for (int x = xMin; x < xMax; x++) {
            int avgIntensity = 0;
            for (int y = yMin; y < yMax; y++) {
                avgIntensity += getIntensity(getPixel(output, x, y));
            }
            avgIntensity /= (yMax - yMin);
            //Apply Threshold Mapping
            if (avgIntensity <= mapToZeroThreshold) {
                avgIntensity = 0;
                line(output, Point(x, yMin), Point(x, yMax), cv::Scalar(0, 0, 0), 1);
            }
            intensityProjection.push_back(avgIntensity);
        }

        //Get length of regions
        Interval interval;
        interval.begin = 0;
        interval.length = 0;
        bool scanning = false;
        for (int i = 0; i < intensityProjection.size(); i++) {
            if (intensityProjection.at(i) != 0) {
                if (!scanning) {
                    interval.begin = i;
                    scanning = true;
                }
                interval.length++;
            } else {
                if (interval.length >= minLength) {
                    intervals.push_back(interval);
                } else {
                    for (int j = 0; j < interval.length; j++) {
                        line(output, Point(xMin + i - j - 1, yMin), Point(xMin + i - j - 1, yMax), cv::Scalar(0, 0, 0),
                             1);
                    }
                }
                interval.begin = 0;
                interval.length = 0;
                scanning = false;
            }
        }
    }

    if (projectionID == HORIZONTAL) {
        //Compute average intensity at each row or column and erase if value below mapToZero threshold
        for (int y = yMin; y < yMax; y++) {
            int avgIntensity = 0;
            for (int x = xMin; x < xMax; x++) {
                avgIntensity += getIntensity(getPixel(output, x, y));
            }
            avgIntensity /= (xMax - xMin);
            //Apply Threshold Mapping
            if (avgIntensity <= mapToZeroThreshold) {
                avgIntensity = 0;
                line(output, Point(xMin, y), Point(xMax, y), cv::Scalar(0, 0, 0), 1);
            }
            intensityProjection.push_back(avgIntensity);
        }

        //Get length of regions
        Interval interval;
        interval.begin = 0;
        interval.length = 0;
        bool scanning = false;
        for (int i = 0; i < intensityProjection.size(); i++) {
            if (intensityProjection.at(i) != 0) {
                if (!scanning) {
                    interval.begin = i;
                    scanning = true;
                }
                interval.length++;
            } else {
                if (interval.length >= minLength) {
                    intervals.push_back(interval);
                } else {
                    for (int j = 0; j < interval.length; j++) {
                        line(output, Point(xMin, yMin + i - j - 1), Point(xMax, yMin + i - j - 1), cv::Scalar(0, 0, 0),
                             1);
                    }
                }
                interval.begin = 0;
                interval.length = 0;
                scanning = false;
            }
        }
    }

    return intervals;
}

double intensityFluctuations(Mat &input, int projectionID) {
    double fluctations = 0;
    if (projectionID == VERTICAL) {
        for (int x = 0; x < input.cols; x++) {
            int intensity = getIntensity(getPixel(input, x, 0));;
            for (int y = 0; y < input.rows; y++) {
                if (intensity != getIntensity(getPixel(input, x, y))) {
                    intensity = getIntensity(getPixel(input, x, y));
                    fluctations++;
                }
            }
        }
        fluctations /= double(input.cols);
    }

    if (projectionID == HORIZONTAL) {
        for (int y = 0; y < input.rows; y++) {
            int intensity = getIntensity(getPixel(input, 0, y));;
            for (int x = 0; x < input.cols; x++) {
                if (intensity != getIntensity(getPixel(input, x, y))) {
                    intensity = getIntensity(getPixel(input, x, y));
                    fluctations++;
                }
            }
        }
        fluctations /= double(input.rows);
    }
    return fluctations;
}

vector<Mat> extractImages(Mat &image, vector<Rect> boundingRects) {
    vector<Mat> images;
    for (int i = 0; i < boundingRects.size(); i++) {
        Mat subImage(image, boundingRects.at(i));
        images.push_back(subImage);
    }
    return images;
}
