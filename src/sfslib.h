#ifndef __SFSLIB_H__
#define __SFSLIB_H__

#include <stdio.h>
#include <math.h>
#include "boost/multi_array.hpp"

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/surface/mls.h>

#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <boost/optional/optional_io.hpp>
#include <optional>
#include <random>
#include <ctime>

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
#include <gsl/gsl_eigen.h>

#include <ginac/ginac.h>

#include <fftw3.h>
#include <complex>
#include <vector>
#include <map>
#include <iterator>
#include "point6d.h"

using namespace std;
using namespace cv;

#define PI 3.1415439

extern double epsilon_zero;
extern int sel_i;
extern int sel_j;
extern const char *keys;

double sqr(double x);

Point3d unit(Point3d p);
double len(Point3d &p);
bool is0(double num, double epsilon);
bool is0(double num);
double normalize0(double x);
double angle(Point3d &a, Point3d &b);
double evalXY(GiNaC::ex ex, GiNaC::symbol x, GiNaC::symbol y, double X,
		double Y);
double sgn(double x);
void print_matrix(const std::string title, const gsl_matrix *m);
void print_matrix(const std::string s, const gsl_matrix *m, bool matlab);

void print_vec(const std::string title, const gsl_vector *v);
void print_vec(const std::string title, const gsl_vector *v, bool matlab);
void print_vec_complex(const std::string title, const gsl_vector_complex *v);
void help();
void gsl_error_handler(const char *reason, const char *file, int line,
		int gsl_errno);
void getPrimeFactors(int n, std::map<int, int> &factors);
void printPrimeFactors(std::map<int, int> &factors);
void gcdPrimeFactors(std::map<int, int> &f1, std::map<int, int> &f2,
		std::map<int, int> &gcd);
int mergePrimeFactors(std::map<int, int> &factors);
int splitPrimeFactors(std::map<int, int> &factors, int aspecth, int aspectw,
		int limit);
string addIndex(std::string s, int idx);
double gsl_matrix_get0(gsl_matrix *m, int row, int col);
Point2i toHemisphericIndex(Point3d l);
Point3d fromHemisphericIndex(Point2i h);
double getParmDouble(String parm, CommandLineParser &parser);
int getParmInt(String parm, CommandLineParser &parser);
int getParmInt(String parm, CommandLineParser &parser);
string getParmString(String parm, CommandLineParser &parser);
double thresh(double x, double t);
void matrix_times_vector(gsl_matrix *m, gsl_vector *v);
gsl_matrix* matrix_times_matrix(gsl_matrix *m0, gsl_matrix *m1);
gsl_matrix* matrix_inverse(gsl_matrix *pi0);
#endif
