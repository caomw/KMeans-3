#pragma once

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <vector>

using namespace cv;
using namespace std;

class KMeans {
private:
	int dimensions;
	int num_clusters;
	Mat_<double> clusters;


public:
	KMeans(int dimensions, int num_clusters);

	void cluster(Mat_<double> samples, Mat_<double>& mu, vector<int>& groups);

private:
	void computeMinMax(Mat_<double> samples, vector<double>& mins, vector<double>& maxs);
};

