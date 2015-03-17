/**
 * マハラノビス距離に基づいた、多変数のK-meansアルゴリズム。
 *
 * @author	Gen Nishida
 * @date	3/17/2015
 * @version	1.0
 */

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "KMeans.h"

using namespace cv;
using namespace std;

int main() {
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

	KMeans kmeans(2, 2);

	Mat_<double> mu;
	vector<int> groups;
	kmeans.cluster(samples, mu, groups);

	cout << mu << endl;
	for (int i = 0; i < groups.size(); ++i) {
		std::cout << groups[i] << std::endl;
	}

	return 0;
}