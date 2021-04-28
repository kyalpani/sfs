#include "output.h"
#include "sfs.h"
#include "sfslib.h"

using namespace std;
using namespace cv;

Output::Output(SFS &sfs_): sfs(sfs_){}

template<class M> void Output::outputCloud(M &m, int type, String pcl_path) {
	outputCloud(m, type, pcl_path, sfs.w_coeffs);
}

template<class M> void Output::outputCloud(M &m, int type, String pcl_path, int w) {
	outputCloud(m, type, pcl_path, sfs.rows, sfs.cols, w);
}

template<class M> void Output::outputCloud(M &m, int type, String pcl_path, int rows, int cols, int w) {
	sfs.markStepAlways(pcl_path);
	pcl::PointCloud<pcl::PointXYZRGB> cloud;
	int start = cloud.width;
	int size = rows * cols + 4; // +4 because of for corner reference points
	cloud.width = start + size;
	cloud.height = 1;
	cloud.is_dense = false;
	cloud.points.resize(size);

	uint8_t rr = 0, gg = 0, bb = 255;
	uint32_t rrgb = ((uint32_t) rr << 16 | (uint32_t) gg << 8 | (uint32_t) bb);

	int idx = 0;
	cloud.points[start + idx].rgb = *reinterpret_cast<float*>(&rrgb);
	cloud.points[start + idx].x = -0.5;
	cloud.points[start + idx].y = 0.5;
	cloud.points[start + idx++].z = 0;

	cloud.points[start + idx].rgb = *reinterpret_cast<float*>(&rrgb);
	cloud.points[start + idx].x = -0.5;
	cloud.points[start + idx].y = -0.5;
	cloud.points[start + idx++].z = 0;

	cloud.points[start + idx].rgb = *reinterpret_cast<float*>(&rrgb);
	cloud.points[start + idx].x = 0.5;
	cloud.points[start + idx].y = -0.5;
	cloud.points[start + idx++].z = 0;

	cloud.points[start + idx].rgb = *reinterpret_cast<float*>(&rrgb);
	cloud.points[start + idx].x = 0.5;
	cloud.points[start + idx].y = 0.5;
	cloud.points[start + idx++].z = 0;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++, idx++) {
			sfs.markSelectedPixel(i, j);

			double X = sfs.getX();
			double Y = sfs.getY();

			if (sfs.cut() || sfs.isBorderPixel()) {
				continue;
			}

			cloud.points[start + idx].x = sfs.cutJ() ? 0.0 : X;
			cloud.points[start + idx].y = sfs.cutI() ? 0.0 : Y;
			cloud.points[start + idx].z = sfs.getter(type, m, i, j);

			uint8_t r = 0, g = 255, b = 0;
			if (w > 0) {
				if (i == w || j == w || i == (rows - w) || j == (cols - w)) {
					r = 255;
					g = 0;
					b = 0;    // Red color
				}
			}

			uint32_t rgb = ((uint32_t) r << 16 | (uint32_t) g << 8 | (uint32_t) b);
			cloud.points[start + idx].rgb = *reinterpret_cast<float*>(&rgb);

			if (sfs.isSelectedPixel()) {
				std::cout << "x=" << cloud.points[start + idx].x << " y=" << cloud.points[start + idx].y << " z=" << cloud.points[start + idx].z << std::endl;
			}
		}
	}
	pcl::io::savePCDFileASCII(String("pcds/").append(pcl_path.append(".pcd")), cloud);
}


struct LightCompare {
	IntMatrix &hemiSphere;

	int quad(const Point2i &p) {
		if (p.x >= 0 && p.x < 90) {
			return 0;
		}
		if (p.x >= 90 && p.x < 180) {
			return 1;
		}
		if (p.x >= 180 && p.x < 270) {
			return 2;
		}
		if (p.x >= 270 && p.x < 360) {
			return 3;
		}
		return 4;
	}

	inline bool operator()(const Point2i &p1, const Point2i &p2) {
		int q1 = quad(p1);
		int q2 = quad(p2);
		if (q1 != q2) {
			return q1 < q2;
		}
		int hits1 = hemiSphere[p1.x][p1.y];
		int hits2 = hemiSphere[p2.x][p2.y];
		return hits1 > hits2;
	}

};

void Output::outputLightCloud(String pcl_path) {
	IntMatrix hemiSphere;

	int az_max = 360;
	int al_max = 180;
	hemiSphere.resize(boost::extents[az_max][al_max]);
	int totalLV = light_vectors.size();
	std::cout << "outputLightCloud=" << totalLV << std::endl;

	pcl::PointCloud<pcl::PointXYZ> cloud;
	cloud.width = totalLV;
	cloud.height = 1;
	cloud.is_dense = false;
	cloud.points.resize(totalLV);

	for (int k = 0; k < az_max; k++) {
		for (int m = 0; m < al_max; m++) {
			hemiSphere[k][m] = 0;
		}
	}

	for (int k = 0; k < totalLV; k++) {
		Point3d l = light_vectors.at(k);
		cloud.points[k].x = l.x;
		cloud.points[k].y = l.y;
		cloud.points[k].z = l.z;
		Point2i hi = toHemisphericIndex(l);
		int kk = hi.x;
		int mm = hi.y;
		if (kk >= 0 && kk < az_max && mm >= 0 && mm < al_max) {
			hemiSphere[kk][mm]++;
		}
	}

	vector<Point2i> hits;
	for (int k = 0; k < az_max; k++) {
		for (int m = 0; m < al_max; m++) {
			if (hemiSphere[k][m] > 10) {
				hits.push_back(Point2i(k, m));
			}
		}
	}
	std::cout << "number of hits=" << hits.size() << std::endl;

	LightCompare lc = { hemiSphere };
	std::sort(hits.begin(), hits.end(), lc);

	std::cout << "number of hit cells=" << hits.size() << std::endl;

	for (int k = 0; k < hits.size(); k++) {
		Point2i hit = hits[k];
		std::cout << "top hit=" << hit << " light=" << fromHemisphericIndex(hit) << " hits=" << hemiSphere[hit.x][hit.y] << std::endl;
	}

	pcl::io::savePCDFileASCII(String("pcds/").append(pcl_path.append(".pcd")), cloud);
}

void Output::outputLight(Point6dMatrix &zMatrix, String pcl_path) {
	int wi = sfs.lw;
	int wj = sfs.lw;

	std::cout << "wi=" << wi << std::endl;
	std::cout << "wj=" << wj << std::endl;

	vector<Point3d> lvs;

	for (int k = -wi; k < wi; k++) {
		for (int m = -wj; m < wj; m++) {
			int i = sfs.li + k;
			int j = sfs.lj + m;
			if (!zMatrix[i][j]) {
				continue;
			}
			cv::Point3d l = sfs.toLight(zMatrix[i][j].value(), i, j);
			std::cout << "angle=" << angle(l, sfs.lv) << std::endl;
			lvs.push_back(l);
			light_vectors.push_back(l);
		}
	}
	/*
	for (int k = 0; k < sfs.rows; k++) {
		for (int m = 0; m < sfs.cols; m++) {
			if (!zMatrix[k][m]) {
				continue;
			}
			cv::Point3d l = sfs.toLight(zMatrix[k][m].value(), k, m);
			std::cout << "angle=" << angle(l, sfs.lv) << std::endl;
			lvs.push_back(l);
			light_vectors.push_back(l);
		}
	}
	*/

	pcl::PointCloud<pcl::PointXYZ> cloud;
	cloud.width = lvs.size();
	cloud.height = 1;
	cloud.is_dense = false;
	cloud.points.resize(cloud.width);

	for (int k = 0; k < cloud.width; k++) {
		Point3d l = lvs.at(k);
		cloud.points[k].x = l.x;
		cloud.points[k].y = l.y;
		cloud.points[k].z = l.z;
	}

	pcl::io::savePCDFileASCII(String("pcds/").append(pcl_path.append(".pcd")), cloud);
}

template<class M> void Output::outputDiffCloud(M &aMatrix, M &bMatrix, int type0, int type, String pcl_path) {
	sfs.markStepAlways(pcl_path);

	double sX = sfs.scaleX(sfs.sj);
	double sY = sfs.scaleY(sfs.si);

	pcl::PointCloud<pcl::PointXYZRGB> cloud;
	int start = cloud.width;
	int size = sfs.rows * sfs.cols;
	cloud.width = start + size;
	cloud.height = 1;
	cloud.is_dense = false;
	cloud.points.resize(size);
	int w = sfs.w_coeffs;
	int idx = 0;
	for (int i = 0; i < sfs.rows; i++) {
		for (int j = 0; j < sfs.cols; j++, idx++) {

			sfs.markSelectedPixel(i, j);

			double X = sfs.getX();
			double Y = sfs.getY();

			if (sfs.cut() || sfs.isBorderPixel()) {
				continue;
			}
			cloud.points[start + idx].x = X;
			cloud.points[start + idx].y = Y;

			double aVal = sfs.getter(type0, aMatrix, i, j);
			double bVal = sfs.getter(type, bMatrix, i, j);

			double res = 1.0;
			if (abs(bVal) > 0) {
				//res = min(abs(aVal - bVal), 1.0);
				//res = min(abs((aVal - bVal) / bVal), 1.0);
				res = min(abs(aVal - bVal) / (1 + abs(bVal)), 1.0);
			}
			cloud.points[start + idx].z = res;

			uint8_t r = 0, g = 255, b = 0;
			if (w > 0) {
				if (i == w || j == w || i == (sfs.rows - w) || j == (sfs.cols - w)) {
					r = 255;
					g = 0;
					b = 0;    // Red color
				}
			}

			uint32_t rgb = ((uint32_t) r << 16 | (uint32_t) g << 8 | (uint32_t) b);
			cloud.points[start + idx].rgb = *reinterpret_cast<float*>(&rgb);

			if (sfs.isSelectedPixel()) {
				std::cout << "x=" << cloud.points[start + idx].x << " y=" << cloud.points[start + idx].y << " z=" << cloud.points[start + idx].z << std::endl;
			}
		}
	}
	pcl::io::savePCDFileASCII(String("pcds/").append(pcl_path.append(".pcd")), cloud);
}

void Output::outputResultClouds(int idx, Point6dMatrix zMatrix, HeightMatrix hMatrix) {

	std::cout << "====================== output results " << idx << " ======================" << std::endl;
	outputCloud < Point6dMatrix > (zMatrix, 8, addIndex("ZZ", idx));
	outputCloud < Point6dMatrix > (zMatrix, sfs.pGetter, addIndex("PP", idx));
	outputCloud < Point6dMatrix > (zMatrix, sfs.qGetter, addIndex("QQ", idx));
	outputCloud < HeightMatrix > (hMatrix, 0, addIndex("H", idx));
	outputLight(zMatrix, addIndex("L", idx));

	if (sfs.generated_derivatives > 0) {
		outputDiffCloud(sfs.hTestMatrix, zMatrix, sfs.pGetter0, sfs.pGetter, addIndex("P_diff", idx));
		outputDiffCloud(sfs.hTestMatrix, zMatrix, sfs.qGetter0, sfs.qGetter, addIndex("Q_diff", idx));
	}

}

void Output::output_shape() {

	outputCloud < Point6dMatrix > (sfs.hTestMatrix, sfs.pGetter0, "P0");
	outputCloud < Point6dMatrix > (sfs.hTestMatrix, sfs.qGetter0, "Q0");
	outputCloud < Point6dMatrix > (sfs.hTestMatrix, 5, "Z0");
	outputCloud < Point6dMatrix > (sfs.sMatrix, 0, "J");
	outputCloud < Point6dMatrix > (sfs.sMatrix, 1, "Jx");
	outputCloud < Point6dMatrix > (sfs.sMatrix, 2, "Jy");

	outputCloud < Point6dMatrix > (sfs.sMatrix, 3, "Jxx");
	outputCloud < Point6dMatrix > (sfs.sMatrix, 4, "Jxy");
	outputCloud < Point6dMatrix > (sfs.sMatrix, 5, "Jyy");

	if (sfs.generated_derivatives != 3) {
		outputCloud < Point6dMatrix > (sfs.uTestMatrix, 0, "U0");
		outputCloud < Point6dMatrix > (sfs.uTestMatrix, 1, "U1");
		outputCloud < Point6dMatrix > (sfs.uTestMatrix, 2, "U2");
		outputCloud < Point6dMatrix > (sfs.uTestMatrix, 3, "U3");
		outputCloud < Point6dMatrix > (sfs.uTestMatrix, 4, "U4");
		outputCloud < Point6dMatrix > (sfs.uTestMatrix, 5, "U5");

		outputCloud < Point6dMatrix > (sfs.mTestMatrix, 0, "M0");
		outputCloud < Point6dMatrix > (sfs.mTestMatrix, 1, "M1");
		outputCloud < Point6dMatrix > (sfs.mTestMatrix, 2, "M2");
		outputCloud < Point6dMatrix > (sfs.mTestMatrix, 3, "M3");
		outputCloud < Point6dMatrix > (sfs.mTestMatrix, 4, "M4");
		outputCloud < Point6dMatrix > (sfs.mTestMatrix, 5, "M5");
	}

	outputCloud < Point6dMatrix > (sfs.sMatrix, 0, "S");
	outputCloud < Point6dMatrix > (sfs.sMatrix, 1, "Sx");
	outputCloud < Point6dMatrix > (sfs.sMatrix, 2, "Sy");

	outputCloud < Point6dMatrix > (sfs.uMatrix, 0, "UU0");
	outputCloud < Point6dMatrix > (sfs.uMatrix, 16, "UU0_sq");
	outputCloud < Point6dMatrix > (sfs.uMatrix, 1, "UU1");
	outputCloud < Point6dMatrix > (sfs.uMatrix, 2, "UU2");
	outputCloud < Point6dMatrix > (sfs.uMatrix, 3, "UU3");
	outputCloud < Point6dMatrix > (sfs.uMatrix, 4, "UU4");
	outputCloud < Point6dMatrix > (sfs.uMatrix, 5, "UU5");

	outputCloud < Point6dMatrix > (sfs.mMatrix, 0, "MM0");
	outputCloud < Point6dMatrix > (sfs.mMatrix, 1, "MM1");
	outputCloud < Point6dMatrix > (sfs.mMatrix, 2, "MM2");
	outputCloud < Point6dMatrix > (sfs.mMatrix, 3, "MM3");
	outputCloud < Point6dMatrix > (sfs.mMatrix, 4, "MM4");
	outputCloud < Point6dMatrix > (sfs.mMatrix, 5, "MM5");

	outputResultClouds(0, sfs.zMatrix0, sfs.hMatrix0);
	outputResultClouds(1, sfs.zMatrix1, sfs.hMatrix1);
	outputResultClouds(2, sfs.zMatrix2, sfs.hMatrix2);
	outputResultClouds(3, sfs.zMatrix3, sfs.hMatrix3);

	outputLightCloud("L");
}

