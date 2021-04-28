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
#include <optional>
#include <boost/optional/optional_io.hpp>
#include <random>
#include <ctime>

#include <gsl/gsl_bspline.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

#include <ginac/ginac.h>

#include <fftw3.h>
#include <complex>
#include <vector>

using namespace std;
using namespace cv;

static inline double sqr(double x) { return x * x;}
inline static double normalize0(double x) {return abs(x) < 0.0001 ? 0 : x;}

static double roundd(double x) {
    return round(x * 100) / 100;
}

class Point6d {

    public:

	double x0;
	double x1;
	double x2;
	double x3;
	double x4;
	double x5;

	Point6d() {
		this->x0 = 0;
		this->x1 = 0;
		this->x2 = 0;
		this->x3 = 0;
		this->x4 = 0;
		this->x5 = 0;
	}

	Point6d(double x0, double x1, double x2, double x3, double x4, double x5) {
		this->x0 = x0;
		this->x1 = x1;
		this->x2 = x2;
		this->x3 = x3;
		this->x4 = x4;
		this->x5 = x5;
	}

	Point6d scale(double factor) {
		return Point6d(this->x0 * factor, this->x1 * factor, this->x2 * factor, this->x3 * factor, this->x4 * factor, this->x5 * factor);
	}

	Point6d sum(Point6d o) {
		return Point6d(this->x0 + o.x0, this->x1 + o.x1, this->x2 + o.x2, this->x3 + o.x3, this->x4 + o.x4, this->x5 + o.x5);
	}

	Point6d diff(Point6d o) {
		return Point6d(this->x0 - o.x0, this->x1 - o.x1, this->x2 - o.x2, this->x3 - o.x3, this->x4 - o.x4, this->x5 - o.x5);
	}

	Point6d round() {
		return Point6d(roundd(this->x0), roundd(this->x1), roundd(this->x2), roundd(this->x3), roundd(this->x4), roundd(this->x5));
	}

	Point6d normalize() {
		return scale(1/sqrt(sqr(this->x0) + sqr(this->x1) + sqr(this->x2) + sqr(this->x3) + sqr(this->x4) + sqr(this->x5)));
	}

	Point6d normalize2() {
		return scale(1 / this->x5);
	}

	double get(int idx) {
	if (idx == 0) {
	    return this->x0;
	}
	if (idx == 1) {
	    return this->x1;
	}
	if (idx == 2) {
	    return this->x2;
	}
	if (idx == 3) {
	    return this->x3;
	}
	if (idx == 4) {
	    return this->x4;
	}
	return this->x5;
	}

	Point6d put(int idx, double val) {
	if (idx == 0) {
	    return Point6d(val, this->x1, this->x2, this->x3, this->x4, this->x5);
	}
	if (idx == 1) {
	    return Point6d(this->x0, val, this->x2, this->x3, this->x4, this->x5);
	}
	if (idx == 2) {
	    return Point6d(this->x0, this->x1, val, this->x3, this->x4, this->x5);
	}
	if (idx == 3) {
	    return Point6d(this->x0, this->x1, this->x2, val, this->x4, this->x5);
	}
	if (idx == 4) {
	    return Point6d(this->x0, this->x1, this->x2, this->x3, val, this->x5);
	}
	return Point6d(this->x0, this->x1, this->x2, this->x3, this->x4, val);
	}

	friend std::ostream& operator<<(std::ostream& os, Point6d const& p) {
	    os << "(" << p.x0 << "," << p.x1 << "," << p.x2 << "," << p.x3 << "," << p.x4 << "," << p.x5 << ")";
	    return os;
	};
};

typedef boost::multi_array<boost::optional<Point3d>, 1> Point3dVector;
typedef boost::multi_array<boost::optional<Point3d>, 2> Point3dMatrix;
typedef boost::multi_array<boost::optional<Point6d>, 2> Point6dMatrix;
typedef boost::multi_array<boost::optional<double>, 2> DoubleMatrix;

const double EPSILON_ZERO = 0.001;
int di=0;
int dj=0;
int dw=0;
int test=0;

double L1, L2, L3;
double test2coeff = 1;

static double scaleX(int j, int cols, int size) {
    return (j - cols / 2) *  1.0 / size;
}

static double scaleY(int i, int rows, int size) {
    return (i - rows / 2) * -1.0 / size;
}

/*
static bool isSelectedPixel(int i, int j) {
     if (i < 0 && j < 0) {
         return true;
     }
     if (i < 0) {
         return j == dj;
     }
     if (j < 0) {
         return i == di;
     }
     return (i == di && j == dj);
}

static bool isSelectedPixel(int i, int j) {
     return (i == di && j == dj) ||
            (dw > 0 && ((i == (di + dw) && j == (dj + dw)) ||
                        (i == (di + dw) && j == (dj - dw)) ||
                        (i == (di - dw) && j == (dj + dw)) ||
                        (i == (di - dw) && j == (dj - dw)) ||
                        (i == di        && j == (dj - dw)) ||
                        (i == di        && j == (dj + dw)) ||
                        (i == (di + dw) && j == dj) ||
                        (i == (di - dw) && j == dj)));
}
*/

static bool isSelectedPixel(int i, int j) {
     return (i == di && j == dj);
}

static void outputMat(Mat & m) {
    for (int k = 0; k < m.rows; k++) {
        for (int l = 0; l < m.cols; l++) {
            //std::cout << roundd(m.at<double>(k, l)) << " ";
            std::cout << m.at<double>(k, l) << " ";
        }
        std::cout << std::endl;
    }
}


int print_matrix(const gsl_matrix *m)
{
        int status, n = 0;

        for (size_t i = 0; i < m->size1; i++) {
                for (size_t j = 0; j < m->size2; j++) {
                        if ((status = printf("%g ", gsl_matrix_get(m, i, j))) < 0)
                                return -1;
                        n += status;
                }

                if ((status = printf("\n")) < 0)
                        return -1;
                n += status;
        }

        return n;
}

void print_vec(const gsl_vector *v) {
    for (size_t i = 0; i < v->size; i++) {
         printf("%g ", gsl_vector_get(v, i));
    }
    printf("\n");
}

void mul_matrix_vec(const gsl_matrix *m, const gsl_vector* b, gsl_vector* out)
{
        int status, n = 0;

        for (size_t i = 0; i < m->size1; i++) {
                double sum = 0;
                for (size_t j = 0; j < m->size2; j++) {

/*
                       printf("%g\n", gsl_matrix_get(m, i, j));
                       printf("%g\n", gsl_vector_get(b, j));
*/
                       sum +=  gsl_matrix_get(m, i, j) * gsl_vector_get(b, j);
                }
                gsl_vector_set(out, (int)i, sum);
        }
}

gsl_matrix * calc_matrix_mul(gsl_matrix *m1, gsl_matrix *m2) {
    // assume both matrices are square;
    size_t sz = m1->size1;
    gsl_matrix *mul = gsl_matrix_alloc(sz, sz);
    for (size_t i = 0; i < sz; i++) {
        for (size_t j = 0; j < sz; j++) {
            double sum = 0;
            gsl_matrix_set(mul, i, j, sum);
            for (size_t k = 0; k < sz; k++) {
                if (i == 1 && j == 0) {
                     printf("a   %g ", gsl_matrix_get(m1, i, k));
                     printf("b   %g ", gsl_matrix_get(m2, k, j));
                }
                sum +=  gsl_matrix_get(m1, i, k) * gsl_matrix_get(m2, k, j);
                if (i == 1 && j == 0) {
                     printf("sum %g\n", sum);
                }
            }
            gsl_matrix_set(mul, i, j, sum);
        }
    }
    return mul;
}



static void gsl_outputMat(gsl_matrix* m) {
    for (int k = 0; k < m->size1; k++) {
        for (int l = 0; l < m->size2; l++) {
            std::cout << gsl_matrix_get (m, k, l) << " ";
        }
        std::cout << std::endl;
    }
}

static void gsl_outputVec(gsl_vector* v) {
    for (int k = 0; k < v->size; k++) {
       std::cout << gsl_vector_get (v, k) << " ";
    }
    std::cout << std::endl;
}

static void outputPixel(std::pair<int, int> p) {
 std::cout << "(" << p.first << "," << p.second << ")";
}

template<class M> static void toMatlabMatrix(M & m, int rows, int cols, std::function<double(M & m, int, int)> getter, String matlabFilename) {
    std::ofstream matlabFile(matlabFilename);

    int idx = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double val = getter(m, i, j);
            matlabFile << val << " ";
        }
        matlabFile << "\n";
    }
}

template<class M> static void toCloud(M & m, pcl::PointCloud<pcl::PointXYZ> & cloud, int rows, int cols, std::function<double(M & m, int, int)> getter) {
    int dim = max(rows, cols);

    int start = cloud.width;
    cloud.width    = start + rows * cols;
    cloud.height   = 1;
    cloud.is_dense = false;
    cloud.points.resize (cloud.width * cloud.height);

/*
    cloud.width    = 1;
    cloud.height   = 1;
    cloud.is_dense = false;
    cloud.points.resize (1);
*/
    int idx = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++, idx++) {

//           cloud.points[start + idx].x = scaleX(j, cols, dim);
//           cloud.points[start + idx].y = scaleY(i, rows, dim);
//           cloud.points[start + idx].z = 0;
//           if (i != 200 && j != 200) {
//                 continue;
//           }
           if (i != 200) {
                 continue;
           }
//            cloud.points[0].x = scaleX(j, cols, dim);
//            cloud.points[0].y = scaleY(i, rows, dim);
//            cloud.points[0].z = getter(m, i, j);

            cloud.points[start + idx].x = scaleX(j, cols, dim);
            cloud.points[start + idx].y = scaleY(i, rows, dim);
            cloud.points[start + idx].z = getter(m, i, j);
        }
    }
}

template<class M> static void toCloud(M & m, pcl::PointCloud<pcl::PointXYZRGB> & cloud, int rows, int cols, std::function<double(M & m, int, int)> getter) {
    int dim = max(rows, cols);

    int start = cloud.width;
    cloud.width    = start + rows * cols;
    cloud.height   = 1;
    cloud.is_dense = false;
    cloud.points.resize (cloud.width * cloud.height);

/*
    cloud.width    = 1;
    cloud.height   = 1;
    cloud.is_dense = false;
    cloud.points.resize (1);
*/
    int idx = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++, idx++) {
//           cloud.points[start + idx].x = scaleX(j, cols, dim);
//           cloud.points[start + idx].y = scaleY(i, rows, dim);
//           cloud.points[start + idx].z = 0;
//           if (i != 200 && j != 200) {
//                 continue;
//           }
//           if (i != 200) {
//                 continue;
//           }
//            cloud.points[0].x = scaleX(j, cols, dim);
//            cloud.points[0].y = scaleY(i, rows, dim);
//            cloud.points[0].z = getter(m, i, j);

            cloud.points[start + idx].x = scaleX(j, cols, dim);
            cloud.points[start + idx].y = scaleY(i, rows, dim);
            cloud.points[start + idx].z = getter(m, i, j);
            uint8_t r = 255, g = 255, b = 255;    // White color
            if (cloud.points[start + idx].z != 0) {
                r = 255;
                g = 0;
                b = 0;    // Red color
            }

            uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
            cloud.points[start + idx].rgb = *reinterpret_cast<float*>(&rgb);
        }
    }
}

template<class M> static void toPQCloud(M & m, pcl::PointCloud<pcl::PointXYZRGBNormal> & cloud, int rows, int cols, std::function<double(M & m, int, int)> pGetter, std::function<double(M & m, int, int)> qGetter) {
    int dim = max(rows, cols);
    int start = cloud.width;
    cloud.width    = start + rows * cols;
    cloud.height   = 1;
    cloud.is_dense = false;
    cloud.points.resize (cloud.width * cloud.height);

    int idx = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++, idx++) {
            cloud.points[start + idx].x = scaleX(j, cols, dim);
            cloud.points[start + idx].y = scaleY(i, rows, dim);
            cloud.points[start + idx].z = 0;

            cloud.points[start + idx].normal[0] = pGetter(m, i, j);
            cloud.points[start + idx].normal[1] = qGetter(m, i, j);
            cloud.points[start + idx].normal[2] = 1;
        }
    }
}

template<class M> static void toCloudExtended(M & m, pcl::PointCloud<pcl::PointXYZ> & cloud, int rows, int cols, int w, std::function<double(M & m, int, int)> getter) {
    int dim = max(rows, cols);

    int start = cloud.width;
    cloud.width    = start + (rows + 2 * w) * (cols + 2 * w);
    cloud.height   = 1;
    cloud.is_dense = false;
    cloud.points.resize (cloud.width * cloud.height);

/*
    cloud.width    = 1;
    cloud.height   = 1;
    cloud.is_dense = false;
    cloud.points.resize (1);
*/
    int idx = 0;
    for (int i = 0; i < (rows + 2 * w); i++) {
        for (int j = 0; j < (cols + 2 * w) ; j++, idx++) {
//           cloud.points[start + idx].x = scaleX(j, cols, dim);
//           cloud.points[start + idx].y = scaleY(i, rows, dim);
//           cloud.points[start + idx].z = 0;
//           if (i != 200 && j != 200) {
//                 continue;
//           }
           if (i != 200) {
                 continue;
           }
//            cloud.points[0].x = scaleX(j, cols, dim);
//            cloud.points[0].y = scaleY(i, rows, dim);
//            cloud.points[0].z = getter(m, i, j);

            cloud.points[start + idx].x = scaleX(j, cols, dim);
            cloud.points[start + idx].y = scaleY(i, rows, dim);
            cloud.points[start + idx].z = getter(m, i - w, j - w);
        }
    }
}


static bool is0(double num, double epsilon) {
    return abs(num) < epsilon;
}

static bool is0(double num) {
    return is0(num, EPSILON_ZERO);
}

static bool leq0(double num) {
    return abs(num) < EPSILON_ZERO || num < 0;
}


static void createSquareImage(Mat & m, int w_smooth, Mat & sq) {
    int rows = m.rows;
    int cols = m.cols;
    Mat gray;
    cvtColor(m, gray, COLOR_BGR2GRAY);

    Mat sq_unblurred;
    sq_unblurred.create(m.size(), CV_16UC1);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int val =  gray.at<uchar>(i, j);
            sq_unblurred.at<unsigned short>(i, j) = val * val;
        }
    }

    blur( sq_unblurred, sq, Size(5, 5));

}


static void addPixel(std::vector<std::pair<int, int>> & pixels, int i, int j) {
     pixels.push_back(std::pair<int, int>(j, i));
}

static void addPixels(std::vector<std::pair<int, int>> & pixels, int i, int j, int w) {
    addPixel(pixels, i + w , j);
    addPixel(pixels, i - w , j);
    addPixel(pixels, i     , j + w);
    addPixel(pixels, i     , j - w);
}

static double evalX(GiNaC::ex ex, GiNaC::symbol x, GiNaC::symbol y, double X, double Y) {
    return GiNaC::ex_to<GiNaC::numeric>(GiNaC::subs(ex, GiNaC::lst{x==X, y==Y})).to_double();
}
static Point3d computeParms(int k, int n_coeffs, std::function<double(int)> getterX, std::function<double(int)> getterY) {

    gsl_matrix * A = gsl_matrix_alloc (2 * n_coeffs + 1, 3);
    gsl_vector* b = gsl_vector_alloc(2 * n_coeffs + 1);
    for (int m = -n_coeffs; m <= n_coeffs; m++) {

        double x = getterX(k + m);

        int mm = n_coeffs + m;

        gsl_matrix_set (A, mm, 0, 1);
        gsl_matrix_set (A, mm, 1, x);
        gsl_matrix_set (A, mm, 2, sqr(x));

        double y = getterY(k + m);
        gsl_vector_set(b, mm, y);
    }

    gsl_matrix * V    = gsl_matrix_alloc (3, 3);
    gsl_vector * S    = gsl_vector_alloc(3);
    gsl_vector * work = gsl_vector_alloc(3);
    gsl_linalg_SV_decomp (A, V, S, work);

    gsl_vector* x = gsl_vector_alloc(3);
    gsl_linalg_SV_solve (A, V, S, b, x);

    double a0 = gsl_vector_get(x, 0);
    double a1 = gsl_vector_get(x, 1);
    double a2 = gsl_vector_get(x, 2);

    Point3d p = Point3d(normalize0(a0) , normalize0(a1) , normalize0(a2));

    return p;

}

void myhandler (const char * reason,
              const char * file,
              int line,
              int gsl_errno) {
   std::cout << "reason=" << reason << " gsl_errno=" << gsl_errno << std::endl;
}

static Point3d computeParms2(int k, int max_points, int n_coeffs, std::function<double(int)> getterX, std::function<double(int)> getterY) {

 //   std::cout << "computeParms2" << std::endl;

    gsl_matrix * A = gsl_matrix_alloc(2 * n_coeffs + 1, 3);
    gsl_vector * b = gsl_vector_alloc(2 * n_coeffs + 1);

    int imin = k - n_coeffs;
    int imax = k + n_coeffs;
    if (imin < 0) {
       imin = 0;
       imax = 2 * n_coeffs + 1;
    } else if (imax >= max_points) {
       imin = max_points - 2 * n_coeffs;
       imax = max_points - 1;
    }

  //  std::cout << "imin=" << imin << " imax=" << imax << std::endl;

    int c = 0;
    for (int m = imin; m < imax; m++, c++) {

        double x = getterX(m);

        gsl_matrix_set (A, c, 0, 1);
        gsl_matrix_set (A, c, 1, x);
        gsl_matrix_set (A, c, 2, sqr(x));

        double y = getterY(m);
        gsl_vector_set(b, c, y);
    }

    gsl_matrix * V    = gsl_matrix_alloc (3, 3);
    gsl_vector * S    = gsl_vector_alloc(3);
    gsl_vector * work = gsl_vector_alloc(3);

    gsl_linalg_SV_decomp (A, V, S, work);
    //std::cout << "3" << std::endl;

    gsl_vector* x = gsl_vector_alloc(3);
    gsl_linalg_SV_solve (A, V, S, b, x);
    //std::cout << "4" << std::endl;
    double a0 = gsl_vector_get(x, 0);
    double a1 = gsl_vector_get(x, 1);
    double a2 = gsl_vector_get(x, 2);

    double x0 = getterX(k);
    Point3d p = Point3d(normalize0(a0) , normalize0(a1) , normalize0(a2));

/*
    std::cout << "A=" << std::endl;
    gsl_outputMat(A);
    std::cout << "V=" << std::endl;
    gsl_outputMat(V);
    std::cout << "S=" << std::endl;
    gsl_outputVec(S);
    std::cout << "work=" << std::endl;
    gsl_outputVec(work);
    std::cout << "b=" << std::endl;
    gsl_outputVec(b);
    std::cout << "p=" << p << std::endl;
*/
    return p;

}

static void calculate_bspline_dir_line(bool xdir, Mat & sqImage, std::function<double(cv::Mat & m, int, int)> getter, int d, size_t n_coeffs, Point3dMatrix & skinMatrix) {

    int rows = sqImage.rows;
    int cols = sqImage.cols;
    int dim  = max(rows, cols);

    int max_points = xdir ? cols : rows;

    vector<double> y_vec;
    for (int k = 0; k < max_points; k++) {
        y_vec.push_back(xdir ? getter(sqImage, d, k) : getter(sqImage, k, d));
    }

    auto getterX = [xdir, max_points, dim](int k) {
        return xdir ? scaleX(k, max_points, dim) : scaleY(k, max_points, dim);
    };

    auto getterY = [max_points, y_vec](int k) {
        return y_vec[k];
    };
/*

      auto getterY = [max_points, y_vec](int k) {

        if (k >= 0 && k < max_points) {
            return y_vec[k];
        }

        if (k < 0) {
            return 2 * y_vec[0] - y_vec[-1 * k];
        }

        return 2 * y_vec[max_points - 1] - y_vec[2 * (max_points - 1) - k];
    };
*/
    for (int k = 0; k < max_points; k++) {
        int ik = xdir ? d : k;
        int jk = xdir ? k : d;
        skinMatrix[ik][jk] = computeParms2(k, max_points, n_coeffs, getterX, getterY);

        if (isSelectedPixel(ik, jk)) {
            std::cout << "xdir=" << xdir << " skin=" << skinMatrix[ik][jk] << std::endl;
        }
    }
}

static void calculate_bspline_dir(bool xdir, Mat & sqImage, std::function<double(cv::Mat & m, int, int)> getter, size_t n_coeffs, Point3dMatrix & skinMatrix) {
    std::cout << "calculate_bspline_dir" << std::endl;
    int max = xdir ? sqImage.rows : sqImage.cols;
    for (int k = 0; k < max; k++) {
        calculate_bspline_dir_line(xdir, sqImage, getter, k, n_coeffs, skinMatrix);
    }
}

static void finalizeSkinMatrix(Mat & sqImage, std::function<double(cv::Mat & m, int, int)> getter, Point3dMatrix & skinMatrixX, Point3dMatrix & skinMatrixY, Point6dMatrix & scMatrix) {
    int rows = sqImage.rows;
    int cols = sqImage.cols;
    int dim  = max(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {

            if (!skinMatrixX[i][j] || !skinMatrixY[i][j]) {
                scMatrix[i][j] = {};
                continue;
            }
            double x = scaleX(j, cols, dim);
            double y = scaleY(i, rows, dim);

            Point3d vx = skinMatrixX[i][j].value();
            Point3d vy = skinMatrixY[i][j].value();

            //double J  = getter(sqImage, i, j);
            double J  = vx.x + vx.y * x + vx.z * x * x;

            scMatrix[i][j] = Point6d(J, vx.y + 2 * vx.z * x, vy.y + 2 * vy.z * y, vx.z / 2 , 0, vy.z / 2);

            if (isSelectedPixel(i, j)) {
                std::cout << "finalizeSkinMatrix" << std::endl;
                std::cout << "x=" << x << " y=" << y << std::endl;
                std::cout << "vx=" << vx << std::endl;
                std::cout << "vy=" << vy << std::endl;
                std::cout << "scMatrix[i][j]=" << scMatrix[i][j] << std::endl;
            }
        }
    }
}

static void smoothSkinMatrixByIndex(int idx, int rows, int cols, int w_smooth, Point6dMatrix & scMatrix) {
    cv::Mat parameterM = cv::Mat::zeros(rows, cols, CV_64F);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (!scMatrix[i][j]) {
                continue;
            }
            Point6d sc = scMatrix[i][j].value();
            parameterM.at<double>(i, j) = sc.get(idx);
        }
    }

    Mat blurred;
    blur( parameterM, blurred, Size(w_smooth, w_smooth));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (!scMatrix[i][j]) {
                continue;
            }
            Point6d sc = scMatrix[i][j].value();
            double val = blurred.at<double>(i, j);
            scMatrix[i][j] = sc.put(idx, val);
        }
    }
}

static void smoothSkinMatrix(int rows, int cols, int w_smooth, Point6dMatrix & scMatrix) {
    smoothSkinMatrixByIndex(1, rows, cols, w_smooth, scMatrix);
    smoothSkinMatrixByIndex(2, rows, cols, w_smooth, scMatrix);
    smoothSkinMatrixByIndex(3, rows, cols, w_smooth, scMatrix);
    smoothSkinMatrixByIndex(5, rows, cols, w_smooth, scMatrix);
}



static void calculate_skin(Mat& sqImage, std::function<double(cv::Mat& m, int, int)> getter, int w_smooth, size_t n_coeffs, Point6dMatrix & scMatrix) {

    int rows = sqImage.rows;
    int cols = sqImage.cols;

    Point3dMatrix skinMatrixX(boost::extents[rows][cols]);
    calculate_bspline_dir(true , sqImage, getter, n_coeffs, skinMatrixX);

    Point3dMatrix skinMatrixY(boost::extents[rows][cols]);
    calculate_bspline_dir(false, sqImage, getter, n_coeffs, skinMatrixY);

    Point3dMatrix skinMatrixXSmooth(boost::extents[rows][cols]);
    Point3dMatrix skinMatrixYSmooth(boost::extents[rows][cols]);

    finalizeSkinMatrix(sqImage, getter, skinMatrixX, skinMatrixY, scMatrix);

    smoothSkinMatrix(rows, cols, w_smooth, scMatrix);
}

static double calc_lhs(Point6d & s, Point6d & u, Point3d & lv) {
    double lm = s.x3 + s.x5;
    double lhs = (s.x0 - sqr(lv.x)) * u.x0 +
                 (s.x0 - sqr(lv.y)) * u.x1 +
                      - lv.x * lv.y * u.x2 +
                               s.x1 * u.x3 +
                               s.x2 * u.x4 +
                                   lm * u.x5;
   return lhs;
}

static void calculate_skin_test(std::function<GiNaC::ex(GiNaC::symbol x, GiNaC::symbol y)> zFunction, int rows, int cols, int dim, Point3d & lv, Point6dMatrix & scMatrix, Point6dMatrix & uTestMatrix, Point3dMatrix & normals) {

    GiNaC::ex l1 = lv.x;
    GiNaC::ex l2 = lv.y;
    GiNaC::ex l3 = lv.z;

    GiNaC::symbol x("x");
    GiNaC::symbol y("y");

    GiNaC::ex z = zFunction(x, y);

    GiNaC::ex p = z.diff(x);
    GiNaC::ex q = z.diff(y);

    GiNaC::ex Px = p.diff(x);
    GiNaC::ex Py = p.diff(y);
    GiNaC::ex Qx = q.diff(x);
    GiNaC::ex Qy = q.diff(y);

    std::ofstream pfile("p_0.txt");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {

            scMatrix[i][j]    = {};


            double X = scaleX(j, cols, dim);
            double Y = scaleY(i, rows, dim);

            try {
            double a0 = evalX(Px  , x, y, X, Y) / 2;
            double a1 = evalX(Qy  , x, y, X, Y) / 2;
            double a2 = evalX(Py  , x, y, X, Y);
            double a3 = evalX(p   , x, y, X, Y);
            double a4 = evalX(q   , x, y, X, Y);
            double a5 = evalX(z   , x, y, X, Y);

            double u0 = 4 * sqr(a0) + sqr(a2);
            double u1 = 4 * sqr(a1) + sqr(a2);
            double u2 = 4 * a2 * (a0 + a1);
            double u3 = 4 * a0 * a3 + 2 * a2 * a4;
            double u4 = 2 * a2 * a3 + 4 * a1 * a4;
            double u5 = sqr(a3) + sqr(a4) + 1;

            uTestMatrix[i][j] = Point6d(u0, u1, u2, u3, u4, u5);

            GiNaC::ex pp = 2 * a0 * (x - X) + a2 * (y - Y) + a3;
            GiNaC::ex qq = 2 * a1 * (y - Y) + a2 * (x - X) + a4;

            //GiNaC::ex pp = 2 * a0 * x + a2 * y + a3;
            //GiNaC::ex qq = 2 * a1 * y + a2 * x + a4;

            //std::cout << "3" << std::endl;

            GiNaC::ex J   = GiNaC::pow(-l1 * pp - l2 * qq + l3, 2) / (GiNaC::pow(pp, 2) + GiNaC::pow(qq, 2) + 1);
            GiNaC::ex Jx  = J.diff(x);
            GiNaC::ex Jy  = J.diff(y);
            GiNaC::ex Jxx = Jx.diff(x);
            GiNaC::ex Jxy = Jx.diff(y);
            GiNaC::ex Jyy = Jy.diff(y);

            //std::cout << "4" << std::endl;

            double J_   = evalX(J , x, y, X, Y);
            //std::cout << "5" << std::endl;

            double Jx_  = evalX(Jx , x, y, X, Y);
            double Jy_  = evalX(Jy , x, y, X, Y);
            double Jxx_ = evalX(Jxx, x, y, X, Y);
            double Jxy_ = evalX(Jxy, x, y, X, Y);
            double Jyy_ = evalX(Jyy, x, y, X, Y);
            Point6d s(J_, Jx_, Jy_, Jxx_ / 2, Jxy_, Jyy_ / 2);

            scMatrix[i][j] = s;
            normals[i][j] = Point3d(-a3, -a4, 1);



            if (isSelectedPixel(i, j)) {
                std::cout << "i="     << i << " j=" << j << std::endl;
                std::cout << "X="     << X << " Y=" << Y << " before" << std::endl;
                std::cout << "z="     <<  evalX(z   , x, y, X, Y) << std::endl;

                std::cout << "p ="     <<  evalX(p   , x, y, X, Y) << std::endl;
                std::cout << "q ="     <<  evalX(q   , x, y, X, Y) << std::endl;

                std::cout << "pp="     <<  evalX(pp   , x, y, X, Y) << std::endl;
                std::cout << "qq="     <<  evalX(qq   , x, y, X, Y) << std::endl;

                std::cout << "px="    <<  evalX(Px  , x, y, X, Y) << std::endl;
                std::cout << "py="    <<  evalX(Py  , x, y, X, Y) << std::endl;
                std::cout << "qx="    <<  evalX(Qx  , x, y, X, Y) << std::endl;
                std::cout << "qy="    <<  evalX(Qy  , x, y, X, Y) << std::endl;

                std::cout << "J="     <<  J_ << std::endl;
                std::cout << "Jx="    <<  Jx_ << std::endl;
                std::cout << "Jy="    <<  Jy_ << std::endl;
                std::cout << "Jxx="   <<  Jxx_ << std::endl;
                std::cout << "Jxy="   <<  Jxy_ << std::endl;
                std::cout << "Jyy="   <<  Jyy_ << std::endl;

                std::cout << "s="     <<  s <<  std::endl;
                std::cout << "a="     <<  Point6d(a0, a1, a2, a3, a4, a5) << std::endl;
                std::cout << "u="     <<  uTestMatrix[i][j].value() << std::endl;
                std::cout << "u n="   <<  uTestMatrix[i][j].value().normalize() << std::endl;

                double lhs = calc_lhs(s, uTestMatrix[i][j].value(), lv);
                std::cout << "lhs=" << lhs << std::endl;

            }



         //   std::cout << "i=" << i << " j=" << j << " s=" << s << std::endl;
            } catch (exception &p) {
                continue;
            }


        }
    }
    pfile.close();
}

static void read_skin_from_file(String scfile_name, Point6dMatrix & scMatrix) {
    std::ifstream scfile(scfile_name);
    std::string line;
    while (std::getline(scfile, line)) {
        std::istringstream iss(line);
        int i, j;
        double sc0, sc1, sc2, sc3, sc4, sc5;

        if (iss >> i >> j >> sc0 >> sc1 >> sc2 >> sc3 >> sc4 >> sc5) {
            scMatrix[i][j] = Point6d(sc0, sc1, sc2, sc3, sc4, sc5);
        } else {
            return;
        }
    }
}

static void generate_skin(Mat & sqImage, int w_smooth, int n_coeffs, bool use_real_data, String sc, int si, int sj, Point6dMatrix & scMatrix, Point3d & lv, Point6dMatrix & uTestMatrix, Point3dMatrix & normals) {
    int rows = sqImage.rows;
    int cols = sqImage.cols;
    int dim  = max(rows, cols);

    pcl::PointCloud<pcl::PointXYZ> cloud;

    auto getterGrayscale = [](Mat& m, int i, int j) {
        double get = m.at<unsigned short>(i, j) / 65025.0 /* 255x255 */;
        //std::cout << "get(" << i << ", " << j << ")=" << get << std::endl;
        return  get;
    };
    toCloud<Mat>(sqImage, cloud, rows, cols, getterGrayscale);
    pcl::io::savePCDFileASCII ("pcds/image.pcd", cloud);

    if (use_real_data) {
        calculate_skin(sqImage, getterGrayscale, w_smooth, n_coeffs, scMatrix);
    } else {
       if (sc.length() > 0) {
           read_skin_from_file(sc, scMatrix);
       } else {
           double sX     = scaleX(sj, cols, dim);
           double sY     = scaleY(si, rows, dim);
           if (test == 0) {
               auto zFunction = [sX, sY, rows, cols, dim](GiNaC::symbol x, GiNaC::symbol y) {
                  // double sX     = scaleX(sj, cols, dim);
                  // double sY     = scaleY(si, rows, dim);
                    //GiNaC::ex z = x * x + y * y;
                    GiNaC::ex z = 2 * pow(x - sX, 2) + pow(y - sY, 2) + 2 * (x - sX) * (y - sY) + (x - sX) + (y - sY);
                    //GiNaC::ex z = pow(x - sX, 2) + pow(y - sY, 2);
                    //GiNaC::ex z = x * x;
                    return z;
               };
               calculate_skin_test(zFunction, rows, cols, dim, lv, scMatrix, uTestMatrix, normals);
           } else if (test == 1) {

               //auto zFunction = [si, sj, rows, cols, dim](GiNaC::symbol x, GiNaC::symbol y) {
               auto zFunction = [sX, sY](GiNaC::symbol x, GiNaC::symbol y) {
                   //GiNaC::ex r2 = sqr(0.5);
                   //GiNaC::ex r2 = sqr(0.7);
                   GiNaC::ex r2 = sqr(1);
                   //GiNaC::ex r2 = sqr(1.5);
                   //GiNaC::ex r2 = sqr(2);
                   //GiNaC::ex a2 = 1;
                   //GiNaC::ex b2 = 1.2;
                   //GiNaC::ex r = 0.707;
                   //GiNaC::ex r = 1.5;
                   //GiNaC::ex r = 2.5;


                   //GiNaC::ex n = GiNaC::pow(x - sX, 2) / a2 + GiNaC::pow(y - sY, 2) / b2;

                   GiNaC::ex z   = GiNaC::sqrt(r2 - GiNaC::pow(x - sX, 2) - GiNaC::pow(y - sY, 2));
                   //GiNaC::ex z   = GiNaC::sqrt(r2 - GiNaC::pow(x - sX, 2) - GiNaC::pow(y - sY, 2));

                   //GiNaC::ex n   = GiNaC::pow(x, 2);
                   //GiNaC::ex z   = -1 * GiNaC::sqrt(GiNaC::pow(r, 2) - n) + r;
                   return z;
               };
               calculate_skin_test(zFunction, rows, cols, dim, lv, scMatrix, uTestMatrix, normals);
           } else if (test == 2) {

               auto zFunction = [sX, sY](GiNaC::symbol x, GiNaC::symbol y) {

                   GiNaC::ex z   = GiNaC::sin(test2coeff * (x - sX)) + GiNaC::cos(test2coeff * (y - sY));
                   return z;
               };
               calculate_skin_test(zFunction, rows, cols, dim, lv, scMatrix, uTestMatrix, normals);
           } else if (test == 3) {
               auto zFunction = [sX, sY](GiNaC::symbol x, GiNaC::symbol y) {
                   GiNaC::ex z   = x * GiNaC::exp(- GiNaC::pow(x - sX, 2) - GiNaC::pow(y - sY, 2));
                   return z;
               };
               calculate_skin_test(zFunction, rows, cols, dim, lv, scMatrix, uTestMatrix, normals);
           }

           std::ofstream scfile("scfiles/scfile.txt");
           for (int i = 0; i < rows; i++) {
               for (int j = 0; j < cols; j++) {
                   double sc0 = 0;
                   double sc1 = 0;
                   double sc2 = 0;
                   double sc3 = 0;
                   double sc4 = 0;
                   double sc5 = 0;
                   boost::optional<Point6d> sc = scMatrix[i][j];
                   if (sc) {
                       Point6d sv = sc.value();
                       sc0 = sv.x0;
                       sc1 = sv.x1;
                       sc2 = sv.x2;
                       sc3 = sv.x3;
                       sc4 = sv.x4;
                       sc5 = sv.x5;
                   }
                   scfile << i << " " << j << " " << " " << sc0 << " " << sc1 << " " << sc2 << " " << sc3 << " " << sc4 << " " << sc5 << std::endl;
               }
           }
       }

       auto getterUTestMatrix = [rows, cols, dim](Point6dMatrix & m, int i, int j) {
           boost::optional<Point6d> uopt = m[i][j];
           if (!uopt) {
               return 0.0;
           }
           Point6d uv = uopt.value();
           double u = uv.x0 * uv.x1 - sqr(uv.x2);
           //std::cout << "u=" << u << std::endl;
           return  u;
       };
    }
}

static Point6d getTransformedCoefficients(Point6d sc, double x, double y) {
   double lambda = sc.x3 + sc.x5;
   return Point6d(normalize0(sc.x0 + 2 * sc.x1 * x + lambda * sqr(x)),
                  normalize0(sc.x0 + 2 * sc.x2 * y + lambda * sqr(y)),
                  normalize0(sc.x1 * y + sc.x2 * x + lambda * x * y),
                  normalize0(sc.x1 + lambda * x),
                  normalize0(sc.x2 + lambda * y),
                  normalize0(lambda)
                 );
}

static Point6d getTransformedCoefficientsU0(Point6d sc, double x, double y) {
   return Point6d(normalize0(sc.x0 + 2 * sc.x1 * x + sc.x3 * sqr(x)),
                  normalize0(                        sc.x3 * sqr(y)),
                  normalize0(            sc.x1 * y + sc.x3 * x * y),
                  normalize0(                sc.x1     + sc.x3 * x),
                  normalize0(                            sc.x3 * y),
                  normalize0(                                sc.x3));
}

static Point6d getTransformedCoefficientsU1(Point6d sc, double x, double y) {
   return Point6d(normalize0(                        sc.x5 * sqr(x)),
                  normalize0(sc.x0 + 2 * sc.x2 * y + sc.x5 * sqr(y)),
                  normalize0(            sc.x2 * x + sc.x5 * x * y),
                  normalize0(                            sc.x5 * x),
                  normalize0(                sc.x2     + sc.x5 * y),
                  normalize0(                                sc.x5));
}

static Point6d getTransformedCoefficientsU2(Point6d sc, double x, double y) {
   return Point6d(normalize0(2 * sc.x0 * x + sc.x1 * sqr(x)),
                  normalize0(                sc.x1 * sqr(y)),
                  normalize0(    sc.x0 * y + sc.x1 * x * y),
                  normalize0(        sc.x0     + sc.x1 * x),
                  normalize0(                    sc.x1 * y),
                  normalize0(                        sc.x1));
}

static Point6d getTransformedCoefficientsU3(Point6d sc, double x, double y) {
   return Point6d(normalize0(                sc.x2 * sqr(x)),
                  normalize0(2 * sc.x0 * y + sc.x2 * sqr(y)),
                  normalize0(    sc.x0 * x + sc.x2 * x * y),
                  normalize0(                    sc.x2 * x),
                  normalize0(        sc.x0     + sc.x2 * y),
                  normalize0(                        sc.x2));
}

Point6d getTranslatedSurfaceCoefficients(Point6d sc, double x0, double y0, double x, double y) {
     double sc_t = sc.x0 + sc.x1 * (x - x0) + sc.x2 * (y - y0) + sc.x3 * sqr(x - x0) + sc.x4 * (x - x0) * (y - y0) + sc.x5 * sqr(y - y0);
     double sc_t_x = sc.x1 + 2 * sc.x3 * (x - x0);
     double sc_t_y = sc.x2 + 2 * sc.x5 * (y - y0);
     return Point6d(sc_t, sc_t_x, sc_t_y, sc.x3, sc.x4, sc.x5);
}


static void setMatrixRow(int k, Point6d vc, gsl_matrix * A) {
    int col = 0;
    gsl_matrix_set (A, k, col++, vc.x0);
    gsl_matrix_set (A, k, col++, vc.x1);
    gsl_matrix_set (A, k, col++, vc.x2);
    gsl_matrix_set (A, k, col++, vc.x3);
    gsl_matrix_set (A, k, col++, vc.x4);
    gsl_matrix_set (A, k, col++, vc.x5);
    gsl_matrix_set (A, k, col++, -1);
  //  gsl_matrix_set (A, k, col++, 0);
}

static void setMatrixRow1(int k, Point6d vc, gsl_matrix * A) {
    int col = 0;
    gsl_matrix_set (A, k, col++, vc.x0);
    gsl_matrix_set (A, k, col++, vc.x1);
    gsl_matrix_set (A, k, col++, vc.x2);
    gsl_matrix_set (A, k, col++, vc.x3);
    gsl_matrix_set (A, k, col++, vc.x4);
    gsl_matrix_set (A, k, col++, vc.x5);
   // gsl_matrix_set (A, k, col++, 0);
    gsl_matrix_set (A, k, col++, 1);
}

static bool fillNormalMatrix(Point6dMatrix& scMatrix, int i, int j, int w, int rows, int cols, int dim, gsl_matrix * A) {


    int im  = i - w;
    int jm  = j - w;

    int ip  = i + w;
    int jp  = j + w;


/*
    boost::optional<Point6d> scmm = scMatrix[i][j];
    boost::optional<Point6d> scm0 = scMatrix[i][j];
    boost::optional<Point6d> scmp = scMatrix[i][j];

    boost::optional<Point6d> sc0m = scMatrix[i][j];
    boost::optional<Point6d> sc00 = scMatrix[i][j];
    boost::optional<Point6d> sc0p = scMatrix[i][j];

    boost::optional<Point6d> scpm = scMatrix[i][j];
    boost::optional<Point6d> scp0 = scMatrix[i][j];
    boost::optional<Point6d> scpp = scMatrix[i][j];
*/
    boost::optional<Point6d> scmm = scMatrix[im][jm];
    boost::optional<Point6d> scm0 = scMatrix[im][j];
    boost::optional<Point6d> scmp = scMatrix[im][jp];

    boost::optional<Point6d> sc0m = scMatrix[i][jm];
    boost::optional<Point6d> sc00 = scMatrix[i][j];
    boost::optional<Point6d> sc0p = scMatrix[i][jp];

    boost::optional<Point6d> scpm = scMatrix[ip][jm];
    boost::optional<Point6d> scp0 = scMatrix[ip][j];
    boost::optional<Point6d> scpp = scMatrix[ip][jp];

    if (!scmm || !scm0 || !scmp || !sc0m || !sc00 || !sc0p || !scpm || !scp0 || !scpp) {
        return {};
    }

    double cmx      = scaleX(jm, cols, dim);
    double cmy      = scaleY(im, cols, dim);

    double cx       = scaleX(j  , cols, dim);
    double cy       = scaleY(i  , rows, dim);

    double cpx      = scaleX(jp, cols, dim);
    double cpy      = scaleY(ip, cols, dim);

    double dxm       = cmx - cx;
    double dxp       = cpx - cx;

    double dym       = cmy - cy;
    double dyp       = cpy - cy;

    cx = 0;
    cy = 0;

    Point6d umm = getTransformedCoefficientsU0(scmm.value(), dxm, dym);
    Point6d um0 = getTransformedCoefficientsU0(scm0.value(), cx , dym);
    Point6d ump = getTransformedCoefficientsU0(scmp.value(), dxp, dym);

    Point6d u0m = getTransformedCoefficientsU0(sc0m.value(), dxm, cy);
    Point6d u00 = getTransformedCoefficientsU0(sc00.value(), cx , cy);
    Point6d u0p = getTransformedCoefficientsU0(sc0p.value(), dxp, cy);

    Point6d upm = getTransformedCoefficientsU0(scpm.value(), dxm, dyp);
    Point6d up0 = getTransformedCoefficientsU0(scp0.value(), cx , dyp);
    Point6d upp = getTransformedCoefficientsU0(scpp.value(), dxp, dyp);

    Point6d vmm = getTransformedCoefficientsU1(scmm.value(), dxm, dym);
    Point6d vm0 = getTransformedCoefficientsU1(scm0.value(), cx , dym);
    Point6d vmp = getTransformedCoefficientsU1(scmp.value(), dxp, dym);

    Point6d v0m = getTransformedCoefficientsU1(sc0m.value(), dxm, cy);
    Point6d v00 = getTransformedCoefficientsU1(sc00.value(), cx , cy);
    Point6d v0p = getTransformedCoefficientsU1(sc0p.value(), dxp, cy);

    Point6d vpm = getTransformedCoefficientsU1(scpm.value(), dxm, dyp);
    Point6d vp0 = getTransformedCoefficientsU1(scp0.value(), cx , dyp);
    Point6d vpp = getTransformedCoefficientsU1(scpp.value(), dxp, dyp);

    if (isSelectedPixel(i, j)) {
         std::cout << "=============" << std::endl;
         std::cout << "=============" << std::endl;
    }
    int k = 0;
    setMatrixRow(k++, umm, A);
    setMatrixRow(k++, um0, A);
    setMatrixRow(k++, ump, A);
    setMatrixRow(k++, u0m, A);
    setMatrixRow(k++, u00, A);
    setMatrixRow(k++, u0p, A);
    setMatrixRow(k++, upm, A);
    setMatrixRow(k++, up0, A);
    setMatrixRow(k++, upp, A);

/*
    setMatrixRow(k++, umm.diff(vmm), A);
    setMatrixRow(k++, um0.diff(vm0), A);
    setMatrixRow(k++, ump.diff(vmp), A);
    setMatrixRow(k++, u0m.diff(v0m), A);
    setMatrixRow(k++, u00.diff(v00), A);
    setMatrixRow(k++, u0p.diff(v0p), A);
    setMatrixRow(k++, upm.diff(vpm), A);
    setMatrixRow(k++, up0.diff(vp0), A);
    setMatrixRow(k++, upp.diff(vpp), A);
*/

/*
    setMatrixRow1(k++, vmm, A);
    setMatrixRow1(k++, vm0, A);
    setMatrixRow1(k++, vmp, A);
    setMatrixRow1(k++, v0m, A);
    setMatrixRow1(k++, v00, A);
    setMatrixRow1(k++, v0p, A);
    setMatrixRow1(k++, vpm, A);
    setMatrixRow1(k++, vp0, A);
    setMatrixRow1(k++, vpp, A);
*/
    return true;
}

static bool fillNormalMatrix2(Point6dMatrix& scMatrix, int i, int j, int w, int rows, int cols, int dim, gsl_matrix * A) {


    int e0i = i;
    int e0j = j - w;

    int e1i = i;
    int e1j = j + w;

    int f0i = i - w;
    int f0j = j;

    int f1i = i + w;
    int f1j = j;

    boost::optional<Point6d> sc   = scMatrix[  i][  j];
/*
    boost::optional<Point6d> sce0 = scMatrix[e0i][e0j];
    boost::optional<Point6d> sce1 = scMatrix[e1i][e1j];
    boost::optional<Point6d> scf0 = scMatrix[f0i][f0j];
    boost::optional<Point6d> scf1 = scMatrix[f1i][f1j];
*/
    boost::optional<Point6d> sce0 = sc;
    boost::optional<Point6d> sce1 = sc;
    boost::optional<Point6d> scf0 = sc;
    boost::optional<Point6d> scf1 = sc;

    if (!sc || !sce0 || !sce1 || !scf0 || !scf1 ) {
        return {};
    }

    double cx       = scaleX(j  , cols, dim);
    double cy       = scaleY(i  , rows, dim);

    double e0x      = scaleX(e0j, cols, dim);
    double e0y      = scaleY(e0i, rows, dim);

    double e1x      = scaleX(e1j, cols, dim);
    double e1y      = scaleY(e1i, rows, dim);

    double f0x      = scaleX(f0j, cols, dim);
    double f0y      = scaleY(f0i, rows, dim);

    double f1x      = scaleX(f1j, cols, dim);
    double f1y      = scaleY(f1i, rows, dim);

    Point6d u0e0 = getTransformedCoefficientsU0(sce0.value(), e0x - cx, e0y - cy);
    Point6d u0e1 = getTransformedCoefficientsU0(sce1.value(), e1x - cx, e1y - cy);

    Point6d u1e0 = getTransformedCoefficientsU1(sce0.value(), e0x - cx, e0y - cy);
    Point6d u1e1 = getTransformedCoefficientsU1(sce1.value(), e1x - cx, e1y - cy);

    Point6d u0f0 = getTransformedCoefficientsU0(scf0.value(), f0x - cx, f0y - cy);
    Point6d u0f1 = getTransformedCoefficientsU0(scf1.value(), f1x - cx, f1y - cy);

    Point6d u1f0 = getTransformedCoefficientsU1(scf0.value(), f0x - cx, f0y - cy);
    Point6d u1f1 = getTransformedCoefficientsU1(scf1.value(), f1x - cx, f1y - cy);

    Point6d u2c  = getTransformedCoefficientsU2(sc.value()  ,   0, 0);
    Point6d u3c  = getTransformedCoefficientsU3(sc.value()  ,   0, 0);

    Point6d u2e0 = getTransformedCoefficientsU2(sce0.value(), e0x - cx, e0y - cy);
    Point6d u3e0 = getTransformedCoefficientsU3(sce0.value(), e0x - cx, e0y - cy);

    Point6d u2e1 = getTransformedCoefficientsU2(sce1.value(), e1x - cx, e1y - cy);
    Point6d u3e1 = getTransformedCoefficientsU3(sce1.value(), e1x - cx, e1y - cy);

    Point6d u2f0 = getTransformedCoefficientsU2(scf0.value(), f0x - cx, f0y - cy);
    Point6d u3f0 = getTransformedCoefficientsU3(scf0.value(), f0x - cx, f0y - cy);

    Point6d u2f1 = getTransformedCoefficientsU2(scf1.value(), f1x - cx, f1y - cy);
    Point6d u3f1 = getTransformedCoefficientsU3(scf1.value(), f1x - cx, f1y - cy);

    if (isSelectedPixel(i, j)) {
         std::cout << "=============" << std::endl;
         std::cout << "u0e0=" << u0e0 << std::endl;
         std::cout << "u0e1=" << u0e1 << std::endl;
         std::cout << "u1e0=" << u1e0 << std::endl;
         std::cout << "u1e1=" << u1e1 << std::endl;

         std::cout << "u0f0=" << u0f0 << std::endl;
         std::cout << "u0f1=" << u0f1 << std::endl;
         std::cout << "u1f0=" << u1f0 << std::endl;
         std::cout << "u1f1=" << u1f1 << std::endl;

         std::cout << "=============" << std::endl;
    }
    int k = 0;
    setMatrixRow(k++, u0e0.diff(u0e1), A);
    setMatrixRow(k++, u1e0.diff(u1e1), A);
    setMatrixRow(k++, u0f0.diff(u0f1), A);
    setMatrixRow(k++, u1f0.diff(u1f1), A);
    setMatrixRow(k++, u2c.scale(2).diff(u2e0).diff(u2e1), A);
    setMatrixRow(k++, u3c.scale(2).diff(u3e0).diff(u3e1), A);
    setMatrixRow(k++, u2c.scale(2).diff(u2f0).diff(u2f1), A);
    setMatrixRow(k++, u3c.scale(2).diff(u3f0).diff(u3f1), A);
    return true;
}

static Point6d calc_a3_a4_aux2(Point6d & sc, Point3d &lv, double a0, double a1, double a2, double u5) {
    double J  = sc.x0;
    double Jx = sc.x1;
    double Jy = sc.x2;

    double l1 = lv.x;
    double l2 = lv.y;

    double K = u5 / J;
    double sqrtK = sqrt(u5 / J);
    double px = 2 * a0;
    double qx = a2;
    double py = a2;
    double qy = 2 * a1;

    double detA = px * qy - py * qx;

    double a3 = (- K / (2 * detA))  * (qy * Jx - qx * Jy) - l1 * sqrtK;
    double a4 = (- K / (2 * detA))  * (px * Jy - py * Jx) - l2 * sqrtK;
//    std::cout << "J=" << J << " Jx=" << Jx << " Jy=" << Jy << " K=" << K << " sqrtK=" << sqrtK << std::endl;
//    std::cout << "a0=" << a0 << " a1=" << a1 << " a2=" << a2 << " a3=" << a3 << " a4=" << a4 << std::endl;

    double I = sqrt(J);
    double Ix = Jx / (2 * I);
    double Iy = Jy / (2 * I);
    double L1 = -sqrt(u5) / detA * (qy * Ix - qx * Iy) - I / u5 * a3;
    double L2 = -sqrt(u5) / detA * (px * Iy - py * Ix) - I / u5 * a4;
//    std::cout << "L1=" << L1 << " L2=" << L2 << " delta=" << sqrt(sqr(L1 - l1) + sqr(L2 - l2)) << std::endl;
    return Point6d(a0, a1, a2, a3, a4, 0);
}


static boost::optional<Point6d> calculate_normal_coefficients(Point6dMatrix& scMatrix, int i, int j, int rows, int cols, int dim, int w, int lastRow) {

    if (i < w || i >= (rows - w) || j < w || j >= (cols - w)) {
        return {};
    }

    int numvars = 7;
    int sz = 9;
    gsl_matrix * A = gsl_matrix_alloc(sz, numvars);
    gsl_matrix * Acopy = gsl_matrix_alloc(sz, numvars);
    gsl_vector * b = gsl_vector_alloc(sz);

    if (!fillNormalMatrix(scMatrix, i, j, w, rows, cols, dim, A)) {
         return {};
    }
    gsl_matrix_memcpy(Acopy, A);

    gsl_matrix * V    = gsl_matrix_alloc(numvars, numvars);
    gsl_vector * S    = gsl_vector_alloc(numvars);
    gsl_vector * work = gsl_vector_alloc(numvars);
    gsl_linalg_SV_decomp (A, V, S, work);
    gsl_vector* X     = gsl_vector_alloc(numvars);
    gsl_linalg_SV_solve (A, V, S, b, X);

    if (isSelectedPixel(i, j)) {
        std::cout << "A=" << std::endl;
        print_matrix(Acopy);
        std::cout << "U=" << std::endl;
        print_matrix(A);
        std::cout << "V=" << std::endl;
        print_matrix(V);
        std::cout << "S=" << std::endl;
        gsl_outputVec(S);
    }

    int row = 0;
    int lastrow = numvars - 1;
    double u0 = gsl_matrix_get(V, row++, lastrow);
    double u1 = gsl_matrix_get(V, row++, lastrow);
    double u2 = gsl_matrix_get(V, row++, lastrow);
    double u3 = gsl_matrix_get(V, row++, lastrow);
    double u4 = gsl_matrix_get(V, row++, lastrow);
    double u5 = gsl_matrix_get(V, row++, lastrow);

    // both u0 and u1 must be positive but
    // because they are a solution of a homogenous system
    // it could be that they are negative, so...
    if (u0 < 0 || u1 < 0) {
        u0 *= -1;
        u1 *= -1;
        u2 *= -1;
        u3 *= -1;
        u4 *= -1;
        u5 *= -1;
    }

    if (u0 * u1 < 0) {
       // std::cout << "2 u(" << i << ", " << j << ")=" << u << std::endl;
        return {};
    }
    Point6d u(u0, u1, u2, u3, u4, u5);
    return u;
}

static void addSolutionsa2zero(Point6d & sc, double u0, double u1, double u3, double u4, double u5, vector<Point6d> & solutions, int i, int j, Point3d &lv) {

     double a0 = sqrt(u0) / 2;
     double a1 = sqrt(u1) / 2;

     double detA = 4 * a0 * a1;
     if (isSelectedPixel(i, j)) {
         std::cout << "detA=" << detA << std::endl;
     }
     if (is0(detA)) {
         // detA cannot be zero
         return;
     }
     double detAInv = 1 / (4 * a0 * a1);

     double a3 = u3 / (4 * a0);
     double a4 = u4 / (4 * a1);

     // calculate homogeneity factor
     double s_inv = u5 - (u1 * u3 * u3 + u0 * u4 * u4) * 0.25 * detAInv * detAInv;
     //   std::cout << "s_inv=" << s_inv << std::endl;

     if (isSelectedPixel(i, j)) {
         std::cout << "s_inv=" << s_inv << std::endl;
     }

     if (s_inv < 0) {
         return;
     }

     double s_root = sqrt(1 / s_inv);

     if (isSelectedPixel(i, j)) {
         std::cout << "s_root=" << s_root << std::endl;
     }

     solutions.push_back(calc_a3_a4_aux2(sc, lv,   a0,  a1,  0, u5).scale(s_root));
     solutions.push_back(calc_a3_a4_aux2(sc, lv,  -a0,  a1,  0, u5).scale(s_root));
     solutions.push_back(calc_a3_a4_aux2(sc, lv,   a0, -a1,  0, u5).scale(s_root));
     solutions.push_back(calc_a3_a4_aux2(sc, lv,  -a0, -a1,  0, u5).scale(s_root));
}

static void addSolutions(Point6d & sc, double u0, double u1, double u2, double u3, double u4, double u5, double discriminant, vector<Point6d> & solutions, int i, int j, Point3d &lv) {

     double a2_squared = 0;
     if (u2 != 0) {
         a2_squared = 0.25 * (u0 + u1 + discriminant) / (1 + pow((u0 - u1) / u2, 2));
         if (isSelectedPixel(i, j)) {
             std::cout << "a2_squared=" << a2_squared << std::endl;
         }

         if (a2_squared < 0) {
             return;
         }
     }

     if (u0 < a2_squared) {
         return;
     }

     double a0 = sqrt(u0 - a2_squared) / 2;

     if (u1 < a2_squared) {
         return;
     }

     double a1 = sqrt(u1 - a2_squared) / 2;
     double a2 = sqrt(a2_squared);

     if (isSelectedPixel(i, j)) {
         std::cout << "a0=" << a0 << " a1=" << a1 << " a2=" << a2 << std::endl;
     }

     double detA = 4 * a0 * a1 - a2_squared;
     if (isSelectedPixel(i, j)) {
         std::cout << "detA=" << detA << std::endl;
     }
     if (is0(detA)) {
         // detA cannot be zero
         return;
     }

     double detAInv = 1 / detA;
     double a3 = detAInv * (a1 * u3 - a2 * u4 / 2);
     double a4 = detAInv * (-a2 * u3 / 2 + a0 * u4);
     //calc_a3_a4_aux2(sc, lv, a0, a1, a2, u5);

     if (isSelectedPixel(i, j)) {
         std::cout << "a3=" << a3 << " a4=" << a4 << std::endl;
     }

     // calculate homogeneity factor
     double s_inv = u5 - (u1 * u3 * u3 - u2 * u3 * u4 + u0 * u4 * u4) * 0.25 * detAInv * detAInv;
     //   std::cout << "s_inv=" << s_inv << std::endl;

     if (isSelectedPixel(i, j)) {
         std::cout << "s_inv=" << s_inv << std::endl;
     }

     if (s_inv < 0) {
         return;
     }

     double s_root = sqrt(1 / s_inv);

     if (isSelectedPixel(i, j)) {
         std::cout << "s_root=" << s_root << std::endl;
     }

     solutions.push_back(calc_a3_a4_aux2(sc, lv,  a0,  a1,  a2, u5).scale(s_root));
     solutions.push_back(calc_a3_a4_aux2(sc, lv, -a0, -a1, -a2, u5).scale(s_root));
}


static void calculate_z_coefficients(Point6dMatrix & scMatrix, boost::optional<Point6d> u, vector<Point6d> & solutions, int i, int j, Point3d &lv) {

    if (!u) {
        return;
    }

    Point6d sc = scMatrix[i][j].value();
    Point6d uu = u.value();
    double u0 = uu.x0;
    double u1 = uu.x1;
    double u2 = uu.x2;
    double u3 = uu.x3;
    double u4 = uu.x4;
    double u5 = uu.x5;

    double u_2_squared = sqr(u2);

    if (isSelectedPixel(i, j)) {
        std::cout << "u_2_squared=" << u_2_squared << " 4 * u0 * u1=" << (4 * u0 * u1) << std::endl;
        std::cout << "u2=" << u2 << " u_2_squared / (4 * u0 * u1)=" << (u_2_squared / (4 * u0 * u1)) << std::endl;
    }

    if (is0(u2) || is0(u_2_squared / (4 * u0 * u1))) {
        addSolutionsa2zero(sc, u0, u1, u3, u4, u5, solutions, i, j, lv);
        return;
    }

    double discriminant = 4 * u0 * u1 - u_2_squared;

    if (isSelectedPixel(i, j)) {
        std::cout << "discrimin=" << discriminant << std::endl;
    }

    if (discriminant < 0) {
        return;
    }

    if (discriminant == 0) {
        addSolutions(sc, u0, u1, u2, u3, u4, u5, 0, solutions, i, j, lv);
    } else {
        double d_root = sqrt(discriminant);
        addSolutions(sc, u0, u1, u2, u3, u4, u5, -d_root, solutions, i, j, lv);
        addSolutions(sc, u0, u1, u2, u3, u4, u5, +d_root, solutions, i, j, lv);
    }
}

//static boost::optional<pcl::PointXYZ> toLight(double x, double y, Point6d solution, Point6d surfaceCoefficients) {
static Point3d toLight(double x, double y, Point6d solution, Point6d surfaceCoefficients) {

   // calculate I, Ix and Iy from related J values
   double I  = sqrt(surfaceCoefficients.x0);
   double Ix = 0.5 * surfaceCoefficients.x1 / I;
   double Iy = 0.5 * surfaceCoefficients.x2 / I;

   // fetch the surface coefficients
   double a0 = solution.x0;
   double a1 = solution.x1;
   double a2 = solution.x2;
   double a3 = solution.x3;
   double a4 = solution.x4;

   // calculate p & q
   double p = a3;
   double q = a4;

   // calculate length of surface normal
   double normn = sqrt(p * p + q * q + 1);

   // calculate magnitude hessian
   double detA = 4 * a0 * a1 - a2 * a2;

   // l1 & l2 by equation 130
   double l1 = -normn / detA * (2 * a1 * Ix - a2 * Iy)      - I * p / normn;
   double l2 = -normn / detA * (   -a2 * Ix + 2  * a0 * Iy) - I * q / normn;

   // l3 by equation 131
   double l3 = I * normn + l1 * p + l2 * q;

   double norml = sqrt(l1 * l1 + l2 * l2 + l3 * l3);
   if (norml > 1.2 || norml < .9) {
       return {};
   }

   return Point3d(l1, l2, l3);

}

static void help()
{
    printf("\nThis sample demonstrates Canny edge detection\n"
           "Call:\n"
           "    /.sfs [image_name -- Default is cat.jpeg]\n\n");
}

const char* keys =
{
    "{help h||}"
    "{@image                      |sfs.img | input image name}"
    "{n_coeffs              count | 10     | number of bspline coefficients}"
    "{w_coeffs              count | 6      | window size for calculation normal coefficients}"
    "{w_smooth              count | 7      | filter window size for smoothing}"
    "{generated_derivatives count | 0      | 0 if derivatives are generated 1 if they are computed from input data }"
    "{l1                          | 0      | x component of light vector [0..1)}"
    "{l2                          | 0      | y component of light vector [0..1)}"
    "{l3                          | 1      | y component of light vector (0..1]}"
    "{li                    count | 0      | the i-index of the image window for computing the light vector }"
    "{lj                    count | 0      | the j-index of the image window for computing the light vector }"
    "{sc                          |        | path to input sc file }"
    "{si                    count | 0      | the i-index of the sample point }"
    "{sj                    count | 0      | the j-index of the sample point }"
    "{lw                    count | 0      | the window size of the image window for computing the light vector }"
    "{di                    count | 0      | the i-index of the debug pixel }"
    "{dj                    count | 0      | the j-index of the debug pixel }"
    "{dw                    count | 0      | the window size of debug pixel window }"
    "{test                  count | 0      | the test number to use for generated derivates }"
    "{lastRow               count | 4      | the row in SVD result to pick for the U vector }"
    "{test2coeff                  | 1      | the coefficient for test 2 }"

};

static double uval(boost::optional<Point6d> u) {
    return u ? (u.value().x0 * u.value().x1 - sqr(u.value().x2)) : 0;
}

double p_(Point6dMatrix & zMatrix, double x, double y, int i, int j) {
    boost::optional<Point6d> sopt = zMatrix[i][j];
    if (sopt) {
        Point6d s = sopt.value();
        //return 2 * -s.x0 * x + s.x2 * y + s.x3;
        return s.x3;
    }
    return 0;
}

double q_(Point6dMatrix & zMatrix, double x, double y, int i, int j) {
    boost::optional<Point6d> sopt = zMatrix[i][j];
    if (sopt) {
        Point6d s = sopt.value();
        //return 2 * s.x1 * y + s.x2 * x + s.x4;
        return s.x4;
    }
    return 0;
}

int main( int argc, const char** argv ) {

    /*
     *    PARSE COMMAND LINE
     */

    gsl_set_error_handler (&myhandler);

    CommandLineParser parser(argc, argv, keys);
    string filename = parser.get<string>(0);
    Mat image = imread(filename, IMREAD_COLOR);
    if(image.empty())
    {
        printf("Cannot read image file: %s\n", filename.c_str());
        help();
        return -1;
    }
    std::cout << "image size:" << image.size() << std::endl;

    int n_coeffs = parser.get<int>("n_coeffs");
    std::cout << "n_coeffs:" << n_coeffs << std::endl;

    int w_coeffs = parser.get<int>("w_coeffs");
    std::cout << "w_coeffs:" << w_coeffs << std::endl;

    int w_smooth = parser.get<int>("w_smooth");
    std::cout << "w_smooth:" << w_smooth << std::endl;

    int generated_derivatives = parser.get<int>("generated_derivatives");
    std::cout << "generated_derivatives:" << generated_derivatives << std::endl;

    String sc = parser.get<String>("sc");
    std::cout << "sc:" << sc << std::endl;

    int si = parser.get<int>("si");
    std::cout << "si:" << si << std::endl;

    int sj = parser.get<int>("sj");
    std::cout << "sj:" << sj << std::endl;

    int li = parser.get<int>("li");
    std::cout << "li:" << li << std::endl;

    int lj = parser.get<int>("lj");
    std::cout << "lj:" << lj << std::endl;

    int lw = parser.get<int>("lw");
    std::cout << "lw:" << lw << std::endl;

    di = parser.get<int>("di");
    std::cout << "di:" << di << std::endl;

    dj = parser.get<int>("dj");
    std::cout << "dj:" << dj << std::endl;

    dw = parser.get<int>("dw");
    std::cout << "dw:" << dw << std::endl;

    test = parser.get<int>("test");
    std::cout << "test:" << test << std::endl;

    int lastRow = parser.get<int>("lastRow");
    std::cout << "lastRow:" << lastRow << std::endl;

    double l1 = parser.get<double>("l1");
    std::cout << "l1:" << l1 << std::endl;
    double l2 = parser.get<double>("l2");
    std::cout << "l2:" << l2 << std::endl;
    double l3 = parser.get<double>("l3");
    std::cout << "l3:" << l3 << std::endl;

    test2coeff = parser.get<double>("test2coeff");
    std::cout << "test2coeff:" << test2coeff << std::endl;

    /*
     *    CALCULATE IMAGE SQUARED (i.e. I * I)
     */

    Mat sqImage;
    createSquareImage(image, w_smooth, sqImage);

    int rows = sqImage.rows;
    int cols = sqImage.cols;
    int dim  = max(rows, cols);

    double normL = sqrt(l1 * l1 + l2 * l2 + l3 * l3);
    L1 = l1 / normL;
    L2 = l2 / normL;
    L3 = l3 / normL;

    Point3d lv(l1 / normL,  l2 / normL,  l3 / normL);

    Point6dMatrix scMatrix(boost::extents[rows][cols]);
    Point6dMatrix uTestMatrix(boost::extents[rows][cols]);
    Point3dMatrix normals_expected(boost::extents[rows][cols]);

    generate_skin(sqImage, w_smooth, n_coeffs, generated_derivatives == 0, sc, si, sj, scMatrix, lv, uTestMatrix, normals_expected);

    /*
     *    CALCULATE LIGHT VECTORS
     */

    Point6dMatrix uMatrix(boost::extents[rows][cols]);
    Point6dMatrix zMatrix(boost::extents[rows][cols]);

    auto getter = [](Mat& m, int i, int j) {
        double get = m.at<unsigned short>(i, j) / 65025.0 /* 255x255 */;
        //std::cout << "get(" << i << ", " << j << ")=" << get << std::endl;
        return  get;
    };

    std::cout << "starting second pass" << std::endl;
    pcl::PointCloud<pcl::PointXYZ> lightVectorCloud;
    lightVectorCloud.height   = 1;
    lightVectorCloud.is_dense = false;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {

             double x = scaleX(j, cols, dim);
             double y = scaleY(i, rows, dim);

             boost::optional<Point6d> u = calculate_normal_coefficients(scMatrix, i, j, rows, cols, dim, w_coeffs, lastRow);
             if (isSelectedPixel(i, j)) {
                std::cout << "i=" << i << " j=" << j << " x=" << x << " y=" << y << " skin1=" << scMatrix[i][j] << std::endl;
             }
             uMatrix[i][j] = u;
             if (!u) {
                 continue;
             }
             if (isSelectedPixel(i, j)) {
                std::cout << " u=" << u.value() << " u n=" << u.value().normalize2() << " uval=" << uval(u) << std::endl;
             }
             vector<Point6d> solutions;
             calculate_z_coefficients(scMatrix, u, solutions, i, j, lv);

             zMatrix[i][j] = {};
             if (solutions.size() > 0) {
                // find the solution with the closest light estimate
                int min_idx = 0;
                double min_diff = std::numeric_limits<double>::max();
                for (int k = 0; k < solutions.size(); k++) {
                    Point3d l = toLight(x, y, solutions[k], scMatrix[i][j].value());
                    double normLL = sqrt(sqr(l.x) + sqr(l.y) + sqr(l.z));
                    double LL1 = l.x / normLL;
                    double LL2 = l.y / normLL;
                    double LL3 = l.z / normLL;

                    double diff = sqrt(sqr(LL1 - L1) + sqr(LL2 - L2) + sqr(LL3 - L3));
                    if (isSelectedPixel(i, j)) {
                        std::cout << "sol=" << solutions[k] << " l=" << l << std::endl;
                    }
                    if (diff < min_diff) {
                        min_diff = diff;
                        min_idx = k;
                    }
                }
                zMatrix[i][j] = solutions[min_idx];
             }
        }
    }
    auto getterSMatrix = [rows, cols, dim](Point6dMatrix & m, int i, int j) {

        boost::optional<Point6d> sopt = m[i][j];
        if (!sopt) {
            return 0.0;
        }
        Point6d s = sopt.value();
        return s.x0;
    };
    auto getterSxMatrix = [rows, cols, dim](Point6dMatrix & m, int i, int j) {
        boost::optional<Point6d> sopt = m[i][j];
        if (!sopt) {
            return 0.0;
        }
        Point6d s = sopt.value();
        return s.x1;
    };
    auto getterSyMatrix = [rows, cols, dim](Point6dMatrix & m, int i, int j) {
        boost::optional<Point6d> sopt = m[i][j];
        if (!sopt) {
            return 0.0;
        }
        Point6d s = sopt.value();
        return s.x2;
    };
    auto getterSxxMatrix = [rows, cols, dim](Point6dMatrix & m, int i, int j) {
        boost::optional<Point6d> sopt = m[i][j];
        if (!sopt) {
            return 0.0;
        }
        Point6d s = sopt.value();
        return s.x3;
    };
    auto getterSyyMatrix = [rows, cols, dim](Point6dMatrix & m, int i, int j) {
        boost::optional<Point6d> sopt = m[i][j];
        if (!sopt) {
            return 0.0;
        }
        Point6d s = sopt.value();
        return s.x5;
    };

    pcl::PointCloud<pcl::PointXYZ> sCloud;
    toCloud<Point6dMatrix>(scMatrix, sCloud, rows, cols, getterSMatrix);
    pcl::io::savePCDFileASCII ("pcds/S.pcd", sCloud);

    Mat s_image;
    s_image.create(rows, cols, CV_16UC1);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {

            double d_val = getterSMatrix(scMatrix, i, j);
            int val =  d_val * 255;
            s_image.at<unsigned short>(i, j) = val;
        }
    }
    std::cout << "writing S" << std::endl;
    imwrite( "pcds/S.jpg", s_image);
    std::cout << "wrote S" << std::endl;


/*
    pcl::PointCloud<pcl::PointXYZ> sCloudExtended;
    toCloudExtended<Point6dMatrix>(scMatrix, sCloudExtended, rows, cols, n_coeffs, getterSMatrixExtended);
    pcl::io::savePCDFileASCII ("pcds/S_ext.pcd", sCloudExtended);

    pcl::PointCloud<pcl::PointXYZ> sxCloudExtended;
    toCloudExtended<Point6dMatrix>(scMatrix, sxCloudExtended, rows, cols, n_coeffs, getterSxMatrixExtended);
    pcl::io::savePCDFileASCII ("pcds/Sx_ext.pcd", sxCloudExtended);

    pcl::PointCloud<pcl::PointXYZ> syCloudExtended;
    toCloudExtended<Point6dMatrix>(scMatrix, syCloudExtended, rows, cols, n_coeffs, getterSyMatrixExtended);
    pcl::io::savePCDFileASCII ("pcds/Sy_ext.pcd", syCloudExtended);
*/
    pcl::PointCloud<pcl::PointXYZ> sxCloud;
    toCloud<Point6dMatrix>(scMatrix, sxCloud, rows, cols, getterSxMatrix);
    pcl::io::savePCDFileASCII ("pcds/Sx.pcd", sxCloud);

    pcl::PointCloud<pcl::PointXYZ> syCloud;
    toCloud<Point6dMatrix>(scMatrix, syCloud, rows, cols, getterSyMatrix);
    pcl::io::savePCDFileASCII ("pcds/Sy.pcd", syCloud);

    pcl::PointCloud<pcl::PointXYZ> sxxCloud;
    toCloud<Point6dMatrix>(scMatrix, sxxCloud, rows, cols, getterSxxMatrix);
    pcl::io::savePCDFileASCII ("pcds/Sxx.pcd", sxCloud);

    pcl::PointCloud<pcl::PointXYZ> syyCloud;
    toCloud<Point6dMatrix>(scMatrix, syyCloud, rows, cols, getterSyyMatrix);
    pcl::io::savePCDFileASCII ("pcds/Syy.pcd", syyCloud);

    auto getterUMatrix = [rows, cols, dim](Point6dMatrix & m, int i, int j) {
        boost::optional<Point6d> uopt = m[i][j];
        if (!uopt) {
            return 0.0;
        }
        Point6d uv = uopt.value();
        double u = uv.x0 * uv.x1 - sqr(uv.x2);
        //double u = uv.x0 * uv.x1;
        //double u = sqr(uv.x2);
        //double u = uv.x0 * uv.x1;
        return  u;
    };

    pcl::PointCloud<pcl::PointXYZRGB> uCloud;
    toCloud<Point6dMatrix>(uMatrix, uCloud, rows, cols, getterUMatrix);
    pcl::io::savePCDFileASCII ("pcds/U.pcd"    , uCloud);

    pcl::PointCloud<pcl::PointXYZRGB> uTestCloud;
    toCloud<Point6dMatrix>(uTestMatrix , uTestCloud, rows, cols, getterUMatrix);
    pcl::io::savePCDFileASCII ("pcds/UTest.pcd"    , uTestCloud);

    auto getterAMatrix = [rows, cols, dim](Point6dMatrix & m, int i, int j) {
        boost::optional<Point6d> sopt = m[i][j];
        if (!sopt) {
            return 0.0;
        }
        Point6d s = sopt.value();
        return s.x0 * s.x1 - sqr(s.x2);
    };

    pcl::PointCloud<pcl::PointXYZRGB> surfaceAcloud;
    toCloud<Point6dMatrix>(zMatrix, surfaceAcloud, rows, cols, getterAMatrix);
    pcl::io::savePCDFileASCII ("pcds/A.pcd", surfaceAcloud);

    auto getterZMatrix = [rows, cols, dim](Point6dMatrix & m, int i, int j) {
        double x = scaleX(j, cols, dim);
        double y = scaleY(i, rows, dim);
        boost::optional<Point6d> sopt = m[i][j];
        if (!sopt) {
            return 0.0;
        }
        Point6d s = sopt.value();
        double z = s.x0 * sqr(x) + s.x1 * sqr(y) + s.x2 * x * y + s.x3 * x + s.x4 * y;
        if (isSelectedPixel(i, j)) {
            std::cout << "i=" << i << " j=" << j << " z=" << z << std::endl;
        }
        if (abs(z) > 1) {
            return 0.0;
        }
        return  z;
    };

    pcl::PointCloud<pcl::PointXYZRGB> surfaceZcloud;
    toCloud<Point6dMatrix>(zMatrix, surfaceZcloud, rows, cols, getterZMatrix);
    pcl::io::savePCDFileASCII ("pcds/Z.pcd", surfaceZcloud);

    auto getterPMatrix = [rows, cols, dim](Point6dMatrix & m, int i, int j) {
        double x = scaleX(j, cols, dim);
        double y = scaleY(i, rows, dim);
        boost::optional<Point6d> sopt = m[i][j];
        if (!sopt) {
            return 0.0;
        }

        Point6d s = sopt.value();
        //double p = 2 * s.x0 * x + s.x2 * y + s.x3;
        double p = s.x3;
        if (isSelectedPixel(i, j)) {
            std::cout << "i=" << i << " j=" << j << " p=" << p << std::endl;
        }
        if (abs(p) > 1) {
            return 0.0;
        }
        return  p;
    };

    auto getterQMatrix = [rows, cols, dim](Point6dMatrix & m, int i, int j) {
        double x = scaleX(j, cols, dim);
        double y = scaleY(i, rows, dim);
        boost::optional<Point6d> sopt = m[i][j];
        if (!sopt) {
            return 0.0;
        }
        Point6d s = sopt.value();
        //double q = 2 * s.x1 * y + s.x2 * x + s.x4;
        double q = s.x4;
        if (isSelectedPixel(i, j)) {
            std::cout << "i=" << i << " j=" << j << " q=" << q << std::endl;
        }
        if (abs(q) > 1) {
            return 0.0;
        }
        return  q;
    };

    toMatlabMatrix<Point6dMatrix>(zMatrix, rows, cols, getterPMatrix, "pcds/P.txt");
    toMatlabMatrix<Point6dMatrix>(zMatrix, rows, cols, getterQMatrix, "pcds/Q.txt");

    pcl::PointCloud<pcl::PointXYZRGB> pCloud;
    toCloud<Point6dMatrix>(zMatrix, pCloud, rows, cols, getterPMatrix);
    pcl::io::savePCDFileASCII ("pcds/P.pcd", pCloud);

    pcl::PointCloud<pcl::PointXYZRGB> qCloud;
    toCloud<Point6dMatrix>(zMatrix, qCloud, rows, cols, getterQMatrix);
    pcl::io::savePCDFileASCII ("pcds/Q.pcd", qCloud);

    pcl::PointCloud<pcl::PointXYZRGBNormal> pqCloud;
    toPQCloud<Point6dMatrix>(zMatrix, pqCloud, rows, cols, getterPMatrix, getterQMatrix);
    pcl::io::savePLYFile ("pcds/pq.ply", pqCloud);

    Mat normalMap;
    normalMap.create(rows, cols, CV_8UC3);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double p = 0;
            double q = 0;
            double x = scaleX(j, cols, dim);
            double y = scaleY(i, rows, dim);

            boost::optional<Point6d> zopt = zMatrix[i][j];
            if (zopt) {
                Point6d z = zopt.value();
                if (isSelectedPixel(i, j)) {
                    std::cout << "z=" << z << std::endl;
                }
                p = 2 * z.x0 * x + z.x2 * y + z.x3;
                q = 2 * z.x1 * y + z.x2 * x + z.x4;
            }
            double len = sqrt(sqr(p) + sqr(q) + 1);
            double r = 255 * (((p / len) + 1) / 2);
            double g = 255 * (((q / len) + 1) / 2);
            double b = 255 * (((1 / len) + 1) / 2);
            if (isSelectedPixel(i, j)) {
                std::cout << "p=" << p << " q=" << q << std::endl;
                std::cout << "len=" << len << " b=" << b << " g=" << g << " r=" << r << std::endl;
            }
            Vec3b color = Vec3b(b, g, r);
            normalMap.at<Vec3b>(Point(i, j)) =  color;
        }
    }

    Mat bgr;
    cvtColor(normalMap, bgr, COLOR_BGR2RGB);
    imwrite( "pcds/normalMap.jpg", bgr );

    Mat hMatrix = cv::Mat::zeros(rows, cols, CV_64F);
    double delta = 1.0 / min(rows, cols);
    std::cout << "delta=" << delta << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {

            if (i == 0 && j == 0) {
                // the height of the upper left corner point is always zero
                continue;
            }
            double p = 0;
            double q = 0;
            double x = scaleX(j, cols, dim);
            double y = scaleY(i, rows, dim);

            if (j > 0) {
                p = p_(zMatrix, x, y, i, j - 1);
            }
            if (i > 0) {
                q = q_(zMatrix, x, y, i - 1, j);
            }
            if (isSelectedPixel(i, j)) {
                std::cout << "x=" << x << " y=" << y << " p=" << p << " q=" << q << " zMatrix=" << zMatrix[i][j] << std::endl;
            }
            if (i == 0 && j > 0) {
                hMatrix.at<double>(i, j) = hMatrix.at<double>(i, j - 1) + p * delta;
                continue;
            }
            if (j == 0 && i > 0) {
                hMatrix.at<double>(i, j) = hMatrix.at<double>(i - 1, j) + q * delta;
                continue;
            }

            double hx = hMatrix.at<double>(i, j - 1) + p * delta;
            double hy = hMatrix.at<double>(i - 1, j) + q * delta;
            if (p == 0) {
                hMatrix.at<double>(i, j) = hy;
                continue;
            }
            if (q == 0) {
                hMatrix.at<double>(i, j) = hx;
                continue;
            }
            hMatrix.at<double>(i, j) = (hx + hy) / 2;
        }
    }

    auto getterHMatrix = [rows, cols, dim](Mat & m, int i, int j) {
        return m.at<double>(i, j);
    };

    pcl::PointCloud<pcl::PointXYZ> surfaceHcloud;
    toCloud<Mat>(hMatrix, surfaceHcloud, rows, cols, getterHMatrix);
    pcl::io::savePCDFileASCII ("pcds/H.pcd", surfaceHcloud);

/*
    pcl::PointCloud<pcl::PointXYZ> sxxCloud;
    toCloud<Point6dMatrix>(scMatrix, sxxCloud, rows, cols, getterSxxMatrix);
    pcl::io::savePCDFileASCII ("pcds/Sxx.pcd", sxxCloud);
    pcl::PointCloud<pcl::PointXYZ> syyCloud;
    toCloud<Point6dMatrix>(scMatrix, syyCloud, rows, cols, getterSyyMatrix);
    pcl::io::savePCDFileASCII ("pcds/Syy.pcd", syyCloud);

    auto getterScMatrix = [rows, cols, dim](Point6dMatrix & m, int i, int j) {
        boost::optional<Point6d> sopt = m[i][j];
        if (!sopt) {
            return 0.0;
        }
        Point6d s = sopt.value();
        return s.x3 * s.x5;// - sqr(s.x2);
    };
    pcl::PointCloud<pcl::PointXYZ> scCloud;
    toCloud<Point6dMatrix>(scMatrix, scCloud, rows, cols, getterScMatrix);
    pcl::io::savePCDFileASCII ("pcds/Sc.pcd", scCloud);

    auto getterZMatrix = [rows, cols, dim](Point6dMatrix & m, int i, int j) {
        double x = scaleX(j, cols, dim);
        double y = scaleY(i, rows, dim);
        boost::optional<Point6d> sopt = m[i][j];
        if (!sopt) {
            return 0.0;
        }
        Point6d s = sopt.value();
        double z = s.x0 * x * x + s.x1 * y * y + s.x2 * x * y + s.x3 * x + s.x4 * y;
        if (isSelectedPixel(i, j)) {
            std::cout << "i=" << i << " j=" << j << " z=" << z << std::endl;
        }
        if (abs(z) > 1) {
            return 0.0;
        }
        return  z;
    };

    pcl::PointCloud<pcl::PointXYZRGB> surfaceZcloud;
    toCloud<Point6dMatrix>(zMatrix, surfaceZcloud, rows, cols, getterZMatrix);
    pcl::io::savePCDFileASCII ("pcds/Z.pcd", surfaceZcloud);

    auto getterVWMatrix = [rows, cols, dim](Point6dMatrix & m, int i, int j) {
        boost::optional<Point6d> uopt = m[i][j];
        if (!uopt) {
            return 0.0;
        }
        Point6d uv = uopt.value();
        //double u = uv.x0 * uv.x1 - sqr(uv.x2);
        double u = sqr(uv.x2);
        return  uv.x5;
    };
*/

}
