/**
 * マハラノビス距離に基づいた、多変数のK-meansアルゴリズム。
 * 実行結果：
 *    [ 0.45,  0.175;
 *     -0.29, -0.133]
 *    0, 0, 1, 0, 1, 1, 0, 1, 1, 1
 * 
 *
 * @author	Gen Nishida
 * @date	3/18/2015
 * @version	1.0
 */

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "KMeans.h"

using namespace cv;
using namespace std;

int main() {

	Mat_<double> samples(1138, 2);
	FILE* fp = fopen("afghanistan_terror_attacks.txt", "r");
	for (int i = 0; i < 1138; ++i) {
		char buff[256];
		fgets(buff, 255, fp);

		double x, y;
		sscanf(buff, "%lf %lf", &x, &y);

		samples(i, 0) = x;
		samples(i, 1) = y;
	}
	fclose(fp);

	cv::Scalar mean1, std_dev1, mean2, std_dev2;
	meanStdDev(samples.col(0), mean1, std_dev1);
	meanStdDev(samples.col(1), mean2, std_dev2);
	Mat_<double> mean = (Mat_<double>(1, 2) << mean1.val[0], mean2.val[0]);
	Mat_<double> std_dev = (Mat_<double>(1, 2) << std_dev1[0], std_dev2[0]);
	for (int r = 0; r < samples.rows; ++r) {
		for (int c = 0; c < samples.cols; ++c) {
			samples(r, c) = (samples(r, c) - mean(0, c)) / std_dev(0, c);
		}
	}

	/*
	Mat_<double> samples = (Mat_<double>(10, 2) << 
		0.9, 0.6,
		0.6, 0.4,
		0.4, 0.8,
		0.4, 0.1,
		0.1, 0.2,
		-0.2, -0.1,
		-0.1, -0.4,
		-0.7, -0.2,
		-0.5, -0.6,
		-0.9, -0.9);
	*/

	KMeans kmeans(2);

	Mat_<double> mu;
	vector<int> groups;
	for (int k = 2; k <= 20; ++k) {
		double aic;
		kmeans.cluster(k, samples, 10, mu, groups, aic);
		//double aic = kmeans.computeAIC(samples, mu, groups);

		printf("%d: AIC=%lf\n", k, aic);
	}

	cout << mu << endl;
	for (int i = 0; i < groups.size(); ++i) {
		std::cout << groups[i] << ",";
	}
	std::cout << std::endl;

	return 0;
}