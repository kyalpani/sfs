#ifndef __SFS_H__
#define __SFS_H__

#include <stdio.h>
#include <math.h>
#include <random>
#include <ctime>
#include <fftw3.h>
#include <complex>
#include <vector>
#include <map>
#include <iterator>
#include <optional>
#include "boost/multi_array.hpp"
#include <boost/optional/optional_io.hpp>

#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <gsl/gsl_bspline.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>
#include <gsl/gsl_statistics_double.h>

#include "point6d.h"
#include "point14d.h"

using namespace std;
using namespace cv;

typedef boost::multi_array<boost::optional<Point3d>, 1> Point3dVector;
typedef boost::multi_array<boost::optional<Point3d>, 2> Point3dMatrix;
typedef boost::multi_array<boost::optional<Point6d>, 2> Point6dMatrix;
typedef boost::multi_array<boost::optional<double>, 2> HeightMatrix;
typedef boost::multi_array<int, 2> IntMatrix;

class SFS {
public :
	string skin;
	int di = 0;
	int dj = 0;
	int dw = 0;
	int test = 0;
	int n_coeffs;
	int w_coeffs;
	int generated_derivatives;
	int si;
	int sj;
	int li;
	int lj;
	int lw;
	int llw;
	double l1, l2, l3;
	double testcoeff1 = 1;
	double testcoeff2 = 1;
	double testcoeff3 = 1;
	double testcoeff4 = 0;
	double testcoeff5 = 0;
	double testcoeff6 = 0;
	String imgFile;
	int rows;
	int cols;
	int dim;
	Point3d lv;
	int selected_i = -1;
	int selected_j = -1;
	Mat I2;
	int cuti;
	int cutj;
	int blocksize;
	int z0height;
	int pGetter0;
	int qGetter0;
	int pGetter;
	int qGetter;
	double cx;
	double cy;

	int aspecth;
	int aspectw;
	int normativeSize;

	int computeLight;
	int getter_upper_limit;
	int lastCol;
	double detLimitLow = 0.1;
	double detLimitHigh = 0.1;
	double epsilon_m0 = 0.1;

	Point6dMatrix sMatrix;
	Point6dMatrix uMatrix;
	Point6dMatrix uTestMatrix;
	Point6dMatrix mMatrix;
	Point6dMatrix mTestMatrix;
	Point6dMatrix hTestMatrix;
	Point6dMatrix tmpMatrix;
	Point6dMatrix pqMatrix;

	Point6dMatrix zMatrix0;
	Point6dMatrix zMatrix1;
	Point6dMatrix zMatrix2;
	Point6dMatrix zMatrix3;

	HeightMatrix hMatrix0;
	HeightMatrix hMatrix1;
	HeightMatrix hMatrix2;
	HeightMatrix hMatrix3;

	Point3dMatrix normals_expected;

	SFS (int argc, const char **argv);
	void recover_shape();

   friend class Output;

private:
	double getter(int type, Point6dMatrix &m, int i, int j);
	double getter(int type, HeightMatrix &m, int i, int j);
	bool isSelectedPixel(int i, int j);
	bool isSelectedPixel();
	void markSelectedPixel(int i, int j);
	void markStep(String step);
	void markStepAlways(String step);
	void makeI2();
	double scaleX(int j);
	double scaleY(int i);
	double getX();
	double getY();
	double getDeltaX();
	double getDeltaY();
	double centerX();
	double centerY();
	double getPixelValue(int i, int j);
	double getSelectedValue();
	bool cut();
	bool cutI();
	bool cutJ();
	bool isBorderPixel();
	void calculate_skin_test(std::function<GiNaC::ex(GiNaC::symbol x, GiNaC::symbol y)> zFunction, std::function<bool(double x, double y)> zDomain);
	void read_skin_from_file(String filename, Point6dMatrix &m);
	void read_skin_from_files();
	void print_test(int ci, int cj, int di, int dj);
	void generate_all_skin_coefficients();
	void generate_skin();
	void collect_neighbors(vector<std::pair<int, int>> &pixels);
	double getGridValue();
	Point14d recover_skin_by_polynomial_fit();
	Point6d calculate_skin_coefficients(Point14d &ss);
	bool fillSkinMatrixRow(int k, std::pair<int, int> &p, double s0, double cx, double cy, gsl_matrix *A, gsl_vector *b);
	void setMatrixRow(int k, Point6d vc, gsl_matrix *A);
	gsl_matrix* getPi(Point6d &c);
	gsl_matrix* getTau(double x, double y);
	gsl_matrix* getA(Point6d &sc, double x, double y);
	void collect_neighborhood_pixels(vector<std::pair<int, int>> &pixels);
	boost::optional<Point6d> getU(gsl_matrix *V, gsl_vector *S, int numvars);
	boost::optional<Point6d> getM(gsl_matrix *V, int numvars);
	bool fillNormalMatrixRow(int k, std::pair<int, int> &p, gsl_matrix *M);
	Point3d toLight(Point6d &h, int i, int j);
	void addSolution(Point6d &u, double h3, double h4, double h5, vector<Point6d> &solutions);
	void addSolutions(Point6d &u, double d_root, vector<Point6d> &solutions);
	void calculate_surface_coefficients();
	boost::optional<Point6d> calculate_normal_coefficients();
	void recover_all_skin_coefficients();
	void recover_all_normal_coefficients();
	void recover_all_surface_coefficients();
	void allocate_height_matrices(int sz, gsl_matrix **A, gsl_matrix **Acopy, gsl_matrix **V, gsl_vector **b, gsl_vector **S, gsl_vector **work, gsl_vector **X);
	void print_height_matrices(gsl_matrix *A, gsl_matrix *Acopy, gsl_matrix *V, gsl_vector *b, gsl_vector *S, gsl_vector *work, gsl_vector *X);
	void free_height_matrices(gsl_matrix *A, gsl_matrix *Acopy, gsl_matrix *V, gsl_vector *b, gsl_vector *S, gsl_vector *work, gsl_vector *X);
	void balance_heights(int h, int w, std::function<double(int, int, int, int)> getRHS, std::function<void(gsl_vector*)> transferHeights);
	void convert_normals_to_height_h0(int k, int m, int h, int w, Point6dMatrix &zMatrix, HeightMatrix &hMatrix);
	void convert_normals_to_height_h1(int k, int m, int h, int w, Point6dMatrix &zMatrix, HeightMatrix &hMatrix);
	void convert_normals_to_height_h0(Point6dMatrix &zMatrix, HeightMatrix &hMatrix);
	void convert_normals_to_height_h1(Point6dMatrix &zMatrix, HeightMatrix &hMatrix);
	void convert_normals_to_height_h2(Point6dMatrix &zMatrix, HeightMatrix &hMatrix);
	void convert_normals_to_height(Point6dMatrix &zMatrix, HeightMatrix &hMatrix);

};
#endif
