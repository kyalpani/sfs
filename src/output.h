#ifndef __OUTPUT_H__
#define __OUTPUT_H__

#include "sfs.h"
#include "sfslib.h"

using namespace std;
using namespace cv;

class Output {

    public:
	Output(SFS & sfs);
	void output_shape();

    private:
	template<class M> void outputCloud(M &m, int type, String pcl_path);
	template<class M> void outputCloud(M &m, int type, String pcl_path, int w);
	template<class M> void outputCloud(M &m, int type, String pcl_path, int rows, int cols, int w);
	template<class M> void outputDiffCloud(M &aMatrix, M &bMatrix, int type0, int type, String pcl_path);
	void outputLightCloud(String pcl_path);
	void outputResultClouds(int idx, Point6dMatrix zMatrix, HeightMatrix hMatrix);
	void outputLight(Point6dMatrix &zMatrix, String pcl_path);

	vector<Point3d> light_vectors;
	SFS & sfs;

};
#endif
