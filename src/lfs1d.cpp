#include "boost/multi_array.hpp"

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/surface/mls.h>

#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <math.h>
#include <stdio.h>
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

using namespace cv;
using namespace std;

const double EPSILON_ZERO = 0.001;
double mm_average = 0;
int mm_count = 0;

int di=0;
int dj=0;
int dw=0;
int test=0;

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

static inline double sqr(double x) {
    return x * x;
}

static double len3d(Point3d p) {
    return sqrt(sqr(p.x) + sqr(p.y) + sqr(p.z));
}

static double roundd(double x) {
    return round(x * 100) / 100;
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

inline static double normalize0(double x) {
    return abs(x) < 0.0001 ? 0 : x;
}

static boost::optional<pcl::PointXYZ> toLight(Point3d l) {
   return pcl::PointXYZ(l.x, l.y, l.z);
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

int print_matrix(const gsl_matrix *m) {
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
        return scale(1 / this->x0);
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
    }
};

typedef boost::multi_array<boost::optional<Point3d>, 1> Point3dVector;
typedef boost::multi_array<boost::optional<Point3d>, 2> Point3dMatrix;
typedef boost::multi_array<boost::optional<Point6d>, 2> Point6dMatrix;
typedef boost::multi_array<boost::optional<double>, 2> DoubleMatrix;


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
//           if (i != 200) {
//                 continue;
//           }
//            cloud.points[0].x = scaleX(j, cols, dim);
//            cloud.points[0].y = scaleY(i, rows, dim);
//            cloud.points[0].z = getter(m, i, j);

            cloud.points[start + idx].x = scaleX(j, cols, dim);
            cloud.points[start + idx].y = scaleY(i, rows, dim);
            cloud.points[start + idx].z = getter(m, i, j);
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
//           if (i != 200) {
//                 continue;
//           }
//            cloud.points[0].x = scaleX(j, cols, dim);
//            cloud.points[0].y = scaleY(i, rows, dim);
//            cloud.points[0].z = getter(m, i, j);

            cloud.points[start + idx].x = scaleX(j, cols, dim);
            cloud.points[start + idx].y = scaleY(i, rows, dim);
            cloud.points[start + idx].z = getter(m, i - w, j - w);
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

static void calculate_skin_test(std::function<GiNaC::ex(GiNaC::symbol x, GiNaC::symbol y)> zFunction, int rows, int cols, int si, int sj, int dim, Point3d & lv, Point6dMatrix & scMatrix, Point6dMatrix & uTestMatrix, Point3dMatrix & normals) {

    GiNaC::ex l1 = lv.x;
    GiNaC::ex l2 = lv.y;
    GiNaC::ex l3 = lv.z;

    GiNaC::symbol x("x");
    GiNaC::symbol y("y");

    //GiNaC::ex z = x * x +  y * y;
    GiNaC::ex z = zFunction(x, y);
    //GiNaC::ex z = x * x + y * y + 0.5 * x * y + 0.6 * x + 0.3 * y;
    //GiNaC::ex z = (x - sX) * (x - sX) + (y - sY) * (y - sY);
    //GiNaC::ex z = x * x + y * y + 0.3 * x * y;

    GiNaC::ex p = z.diff(x);
    GiNaC::ex q = z.diff(y);

    GiNaC::ex Px = p.diff(x);
    GiNaC::ex Py = p.diff(y);
    GiNaC::ex Qx = q.diff(x);
    GiNaC::ex Qy = q.diff(y);

    GiNaC::ex I = (-l1 * p - l2 * q + l3) / GiNaC::pow(GiNaC::pow(p, 2) + GiNaC::pow(q, 2) + 1, 0.5);
    GiNaC::ex Ix = I.diff(x);
    GiNaC::ex Iy = I.diff(y);
    GiNaC::ex J = I * I;
    GiNaC::ex Jx = J.diff(x);
    GiNaC::ex Jy = J.diff(y);
    GiNaC::ex Jxx = Jx.diff(x);
    GiNaC::ex Jxy = Jx.diff(y);
    GiNaC::ex Jyy = Jy.diff(y);

    //std::cout << "Jx=" << Jx << std::endl;

    std::ofstream pfile("p_0.txt");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {

            scMatrix[i][j]    = {};

            double X = scaleX(j, cols, dim);
            double Y = scaleY(i, rows, dim);

            double I_   = 0;
            double J_   = 0;
            try {
                I_ = evalX(I  , x, y, X, Y);
                J_ = evalX(J  , x, y, X, Y);
            } catch (exception &p) {
                continue;
            }
            double Ix_  = evalX(Ix , x, y, X, Y);
            double Iy_  = evalX(Iy , x, y, X, Y);

            double Jx_  = evalX(Jx , x, y, X, Y);
            double Jy_  = evalX(Jy , x, y, X, Y);
            double Jxx_ = evalX(Jxx, x, y, X, Y);
            double Jxy_ = evalX(Jxy, x, y, X, Y);
            double Jyy_ = evalX(Jyy, x, y, X, Y);
            Point6d s(J_, Jx_, Jy_, Jxx_ / 2, Jxy_, Jyy_ / 2);

            scMatrix[i][j] = s;

            double a0 = evalX(Px  , x, y, X, Y) / 2;
            double a1 = evalX(Qy  , x, y, X, Y) / 2;
            double a2 = evalX(Py  , x, y, X, Y);
            double a3 = evalX(p   , x, y, X, Y) - 2 * a0 * X - a2 * Y;
            double a4 = evalX(q   , x, y, X, Y) - 2 * a1 * Y - a2 * X;
            double a5 = evalX(z   , x, y, X, Y);

            normals[i][j] = Point3d(-a3, -a4, 1);

            double u0 = 4 * sqr(a0) + sqr(a2);
            double u1 = 4 * sqr(a1) + sqr(a2);
            double u2 = 4 * a2 * (a0 + a1);
            double u3 = 4 * a0 * a3 + 2 * a2 * a4;
            double u4 = 2 * a2 * a3 + 4 * a1 * a4;
            double u5 = sqr(a3) + sqr(a4) + 1;

            uTestMatrix[i][j] = Point6d(u0, u1, u2, u3, u4, u5);


            if (isSelectedPixel(i, j)) {
                std::cout << "i="     << i << " j=" << j << std::endl;
                std::cout << "X="     << X << " Y=" << Y << " before" << std::endl;
                std::cout << "z="     <<  evalX(z   , x, y, X, Y) << std::endl;

                std::cout << "p="     <<  evalX(p   , x, y, X, Y) << std::endl;
                std::cout << "q="     <<  evalX(q   , x, y, X, Y) << std::endl;

                std::cout << "px="    <<  evalX(Px  , x, y, X, Y) << std::endl;
                std::cout << "py="    <<  evalX(Py  , x, y, X, Y) << std::endl;
                std::cout << "qx="    <<  evalX(Qx  , x, y, X, Y) << std::endl;
                std::cout << "qy="    <<  evalX(Qy  , x, y, X, Y) << std::endl;

                std::cout << "I="     <<  evalX(I   , x, y, X, Y) << std::endl;
                std::cout << "Ix="    <<  evalX(Ix  , x, y, X, Y) << std::endl;
                std::cout << "Iy="    <<  evalX(Iy  , x, y, X, Y) << std::endl;

                std::cout << "J="     <<  evalX(J   , x, y, X, Y) << std::endl;
                std::cout << "Jx="    <<  evalX(Jx  , x, y, X, Y) << std::endl;
                std::cout << "Jy="    <<  evalX(Jy  , x, y, X, Y) << std::endl;
                std::cout << "Jxx="   <<  evalX(Jxx  , x, y, X, Y) << std::endl;
                std::cout << "Jxy="   <<  evalX(Jxy  , x, y, X, Y) << std::endl;
                std::cout << "Jyy="   <<  evalX(Jyy  , x, y, X, Y) << std::endl;

/*
                std::cout << "a0="    <<  a0 << std::endl;
                std::cout << "a1="    <<  a1 << std::endl;
                std::cout << "a2="    <<  a2 << std::endl;
                std::cout << "a3="    <<  a3 << std::endl;
                std::cout << "a4="    <<  a4 << std::endl;
                std::cout << "a5="    <<  a5 << std::endl;
*/
                std::cout << "s="     <<  s <<  std::endl;
                std::cout << "a="     <<  Point6d(a0, a1, a2, a3, a4, a5) << std::endl;
                std::cout << "u="     <<  uTestMatrix[i][j].value() << std::endl;
                std::cout << "u n="   <<  uTestMatrix[i][j].value().normalize2() << std::endl;
            }

            

         //   std::cout << "i=" << i << " j=" << j << " s=" << s << std::endl;

        }
    }
    pfile.close();
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

    gsl_matrix * V    = gsl_matrix_alloc(3, 3);
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
           if (test == 0) {
               auto zFunction = [](GiNaC::symbol x, GiNaC::symbol y) {
                    //GiNaC::ex z = x * x + y * y + x * y;
                    GiNaC::ex z = x * x;
                    return z;
               };
               calculate_skin_test(zFunction, rows, cols, si, sj, dim, lv, scMatrix, uTestMatrix, normals);
           } else {

               auto zFunction = [](GiNaC::symbol x, GiNaC::symbol y) {
                   GiNaC::ex r = 0.5;
                   //GiNaC::ex r = 0.707;
                   //GiNaC::ex r = 1.5;


                   //double sX     = scaleX(sj, cols, dim);
                   //double sY     = scaleY(si, rows, dim);
                   //GiNaC::ex n = GiNaC::pow(x - sX, 2) + GiNaC::pow(y - sY, 2);

                   //GiNaC::ex n   = GiNaC::pow(x, 2) + GiNaC::pow(y, 2);
                   //GiNaC::ex z   = -1 * GiNaC::sqrt(GiNaC::pow(r, 2) - n) + r;

                   GiNaC::ex n   = GiNaC::pow(x, 2);
                   GiNaC::ex z   = -1 * GiNaC::sqrt(GiNaC::pow(r, 2) - n) + r;
                   return z;
               };
               calculate_skin_test(zFunction, rows, cols, si, sj, dim, lv, scMatrix, uTestMatrix, normals);
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
static Point6d getTransformedCoefficients(Point6d sc, double x, double y, Point3d & lv) {
   double lambda = sc.x3 + sc.x5;
   return Point6d(normalize0(sc.x0 + 2 * sc.x1 * x + lambda * sqr(x) - sqr(lv.x)),
                  normalize0(sc.x0 + 2 * sc.x2 * y + lambda * sqr(y) - sqr(lv.y)),
                  normalize0(sc.x1 * y + sc.x2 * x + lambda * x * y - lv.x * lv.y),
                  normalize0(sc.x1 + lambda * x),
                  normalize0(sc.x2 + lambda * y),
                  normalize0(lambda)
                 );
/*
*/
/*
   return Point6d(normalize0( sc.x0 - 2 * sc.x1 * x + lambda * x * x - lv.x * lv.x),
                  normalize0( sc.x0 - 2 * sc.x2 * y + lambda * y * y - lv.y * lv.y),
                  normalize0(-sc.x1 * y - sc.x2 * x + lambda * x * y - lv.x * lv.y),
                  normalize0( sc.x1 - lambda * x),
                  normalize0( sc.x2 - lambda * y),
                  normalize0(lambda)
                 );
*/
}

static Point6d getTransformedCoefficients4(Point6d sc, double x, double y) {
   double lambda = sc.x3 + sc.x5;
   return Point6d(normalize0(sc.x0 + 2 * sc.x1 * x + lambda * sqr(x)),
                  normalize0(sc.x0 + 2 * sc.x2 * y + lambda * sqr(y)),
                  normalize0(sc.x1 * y + sc.x2 * x + lambda * x * y),
                  normalize0(sc.x1 + lambda * x),
                  normalize0(sc.x2 + lambda * y),
                  normalize0(lambda)
                 );
}

static Point6d getTransformedCoefficientsV(Point6d sc, double x, double y) {
   return Point6d(normalize0(2 * sc.x0 * x + sc.x1 * x * x),
                  normalize0(                sc.x1 * y * y),
                  normalize0(    sc.x0 * y + sc.x1 * x * y),
                  normalize0(    sc.x0     + sc.x1 * x),
                  normalize0(    sc.x1 * y),
                  normalize0(    sc.x1)
                 );
}
 
static Point6d getTransformedCoefficientsW(Point6d sc, double x, double y) {
   return Point6d(normalize0(                sc.x2 * x * x),
                  normalize0(2 * sc.x0 * y + sc.x2 * y * y),
                  normalize0(    sc.x0 * x + sc.x2 * x * y),
                  normalize0(    sc.x2 * x),
                  normalize0(    sc.x0     + sc.x2 * y),
                  normalize0(    sc.x2)
                 );
}

static bool fillRowGsl(int k, Point6dMatrix& scMatrix, int i, int j, std::pair<int, int> & p, int rows, int cols, int dim, Point3d & lightVector, gsl_matrix * A,  gsl_vector* b) {

    boost::optional<Point6d> sc0 = scMatrix[i][j];

    int ii = p.second;
    int jj = p.first;
    boost::optional<Point6d> sc = scMatrix[ii][jj];

    if (!sc0 || !sc) {
        return {};
    }

    double x    = scaleX(j, cols, dim);
    double y    = scaleY(i, rows, dim);

    double xx    = scaleX(jj, cols, dim);
    double yy    = scaleY(ii, rows, dim);

    //Point6d vc  = getTransformedCoefficients(sc.value(), 0, 0, lightVector);
    //Point6d vc  = getTransformedCoefficients(sc.value(), xx - x, yy - y, lightVector);
    Point6d vc0  = getTransformedCoefficients(sc0.value(), x ,  y, lightVector);
    Point6d vc   = getTransformedCoefficients(sc0.value(), xx, yy, lightVector);

    if (isSelectedPixel(i, j)) {
         std::cout << "=============" << std::endl;
         std::cout << "sc=" << sc << std::endl;
         std::cout << "vc=" << vc << std::endl;
         std::cout << "=============" << std::endl;
    }


/*
    gsl_matrix_set (A, k, 0, vc.diff(vc0).x0);
    gsl_matrix_set (A, k, 1, vc.diff(vc0).x1);
    gsl_matrix_set (A, k, 2, vc.diff(vc0).x2);
    gsl_matrix_set (A, k, 3, vc.diff(vc0).x3);
    gsl_matrix_set (A, k, 4, vc.diff(vc0).x4);
    gsl_matrix_set (A, k, 5, vc.diff(vc0).x5);
    gsl_vector_set (b, k, -vc.diff(vc0).x5);
*/

    gsl_matrix_set (A, k, 0, vc.x0);
    gsl_matrix_set (A, k, 1, vc.x1);
    gsl_matrix_set (A, k, 2, vc.x2);
    gsl_matrix_set (A, k, 3, vc.x3);
    gsl_matrix_set (A, k, 4, vc.x4);
    gsl_matrix_set (A, k, 5, vc.x5);
    gsl_vector_set (b, k, -vc.x5);

    return true;
}

static bool fillRowGsl1(int k, Point6dMatrix& scMatrix, int i, int j, std::pair<int, int> & p,  int rows, int cols, int dim, gsl_matrix * A,  gsl_vector* b, Point3d & lightVector) {

    int ii = p.second;
    int jj = p.first;

    boost::optional<Point6d> sc0 = scMatrix[i][j];
    boost::optional<Point6d> sc  = scMatrix[ii][jj];

    if (!sc0 || !sc) {
        return {};
    }

    double x    = scaleX(j, cols, dim);
    double y    = scaleY(i, rows, dim);

    double xx    = scaleX(jj, cols, dim);
    double yy    = scaleY(ii, rows, dim);

    Point6d vc0  = getTransformedCoefficients(sc0.value(), x ,  y, lightVector);
    Point6d vc   = getTransformedCoefficients(sc0.value(), xx, yy, lightVector);

    if (isSelectedPixel(i, j)) {
         std::cout << "=============" << std::endl;
         std::cout << "sc=" << sc << std::endl;
         std::cout << "vc=" << vc << std::endl;
         std::cout << "=============" << std::endl;
    }

    vc = vc.diff(vc0);
    gsl_matrix_set (A, k, 0, vc.x0);
    gsl_matrix_set (A, k, 1, vc.x1);
    gsl_matrix_set (A, k, 2, vc.x2);
    gsl_matrix_set (A, k, 3, vc.x3);
    gsl_matrix_set (A, k, 4, vc.x4);
   // gsl_matrix_set (A, k, 5, vc.x5);
    gsl_vector_set (b, k, -vc.x5);

    return true;
}

static bool fillRowGsl2(int k, Point6dMatrix& scMatrix, int i, int j, int i1, int j1, int i2, int j2, int rows, int cols, int dim, gsl_matrix * A,  gsl_vector* b, Point3d & lightVector) {

    boost::optional<Point6d> sc = scMatrix[i][j];

    if (!sc) {
        return {};
    }

    double x    = scaleX(j1, cols, dim);
    double y    = scaleY(i1, rows, dim);

    double xx    = scaleX(j2, cols, dim);
    double yy    = scaleY(i2, rows, dim);

    Point6d vc1  = getTransformedCoefficients(sc.value(), x ,  y, lightVector);
    Point6d vc2  = getTransformedCoefficients(sc.value(), xx, yy, lightVector);

    if (isSelectedPixel(i, j)) {
         std::cout << "=============" << std::endl;
         std::cout << "sc=" << sc << std::endl;
         std::cout << "vc1=" << vc1 << std::endl;
         std::cout << "vc2=" << vc2 << std::endl;
         std::cout << "=============" << std::endl;
    }

    Point6d vc = vc1.diff(vc2);
    gsl_matrix_set (A, k, 0, vc.x0);
    gsl_matrix_set (A, k, 1, vc.x1);
    gsl_matrix_set (A, k, 2, vc.x2);
    gsl_matrix_set (A, k, 3, vc.x3);
    gsl_matrix_set (A, k, 4, vc.x4);
   // gsl_matrix_set (A, k, 5, vc.x5);
    gsl_vector_set (b, k, -vc.x5);

    return true;
}


static bool fillRowGsl4(int k, Point6dMatrix& scMatrix, int i, int j, std::pair<int, int> & p,  int rows, int cols, int dim, gsl_matrix * A,  gsl_vector* b) {



    double c_x   = scaleX(j, cols, dim);
    double c_y   = scaleY(i, rows, dim);
    boost::optional<Point6d> sc_c = scMatrix[i][j];

    double d_x = scaleX(p.first , cols, dim);
    double d_y = scaleY(p.second, rows, dim);
    boost::optional<Point6d> sc_d = scMatrix[p.second][p.first];

    int jj = 2 * j - p.first;
    int ii = 2 * i - p.second;
    double e_x = scaleX(jj, cols, dim);
    double e_y = scaleY(ii, rows, dim);
    boost::optional<Point6d> sc_e = scMatrix[ii][jj];

    if (!sc_c || !sc_d || !sc_e) {
        return {};
    }

    Point6d vc_0 = getTransformedCoefficientsV(sc_c.value(), c_x, c_y);
    Point6d vc_d = getTransformedCoefficientsV(sc_d.value(), d_x, d_y);
    Point6d vc_e = getTransformedCoefficientsV(sc_e.value(), e_x, e_y);
    Point6d vc = vc_d.sum(vc_e).diff(vc_0.scale(2));
 
    Point6d wc_0 = getTransformedCoefficientsW(sc_c.value(), c_x, c_y);
    Point6d wc_d = getTransformedCoefficientsW(sc_d.value(), d_x, d_y);
    Point6d wc_e = getTransformedCoefficientsW(sc_e.value(), e_x, e_y);
    Point6d wc = wc_d.sum(wc_e).diff(wc_0.scale(2));

    if (isSelectedPixel(i, j)) {
         std::cout << "=============" << std::endl;
         std::cout << "p.second=" << p.second << " p.first=" << p.first << std::endl;
         std::cout << "sc_c=  " << sc_c << std::endl;
         std::cout << "=============" << std::endl;
    }

    gsl_matrix_set (A, k, 0, vc.x0);
    gsl_matrix_set (A, k, 1, vc.x1);
    gsl_matrix_set (A, k, 2, vc.x2);
    gsl_matrix_set (A, k, 3, vc.x3);
    gsl_matrix_set (A, k, 4, vc.x4);
    gsl_matrix_set (A, k, 5, vc.x5);
    gsl_vector_set (b, k, -vc.x5);

    gsl_matrix_set (A, k + 1, 0, wc.x0);
    gsl_matrix_set (A, k + 1, 1, wc.x1);
    gsl_matrix_set (A, k + 1, 2, wc.x2);
    gsl_matrix_set (A, k + 1, 3, wc.x3);
    gsl_matrix_set (A, k + 1, 4, wc.x4);
    gsl_matrix_set (A, k + 1, 5, wc.x5);
    gsl_vector_set (b, k + 1, -wc.x5);

    return true;
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
     double sc_t_x = sc.x1 + sc.x3 * (x - x0);
     double sc_t_y = sc.x2 + sc.x5 * (y - y0);
     return Point6d(sc_t, sc_t_x, sc_t_y, sc.x3, sc.x4, sc.x5);
}

static bool fillRowGsl6(int k, Point6dMatrix& scMatrix, int i, int j, std::pair<int, int> & p, int rows, int cols, int dim, gsl_matrix * A,  gsl_vector* b) {


    if (!scMatrix[i][j]) { 
        return {}; 
    }

    int i1 = p.second;
    int j1 = p.first;

    int di = (i - p.second) * 1.5;
    int dj = (j - p.first) * 1.5;

    //int i2 = i1 + di;
    //int j2 = j1 + dj;

    int i2 = 2 * i - i1;
    int j2 = 2 * j - j1;

    if (!scMatrix[i1][j1] || !scMatrix[i2][j2]) { return {}; }

   Point6d sc1  = scMatrix[i1][j1].value();
   Point6d sc2  = scMatrix[i2][j2].value();

    double x0    = scaleX(j , cols, dim);
    double y0    = scaleY(i , rows, dim);

    double x1    = scaleX(j1, cols, dim);
    double y1    = scaleY(i1, rows, dim);

    double x2    = scaleX(j2, cols, dim);
    double y2    = scaleY(i2, rows, dim);

    Point6d sc   = scMatrix[i][j].value();

    //Point6d sc1  = getTranslatedSurfaceCoefficients(sc, x0, y0, x1, y1);
    //Point6d sc2  = getTranslatedSurfaceCoefficients(sc, x0, y0, x2, y2);

    //Point6d sc1  = sc;
    //Point6d sc2  = sc;

    double dx    = x2 - x1;
    double dy    = y2 - y1;

    Point6d u0c  = getTransformedCoefficientsU0(sc1, x1 - x0, y1 - y0);
    Point6d u1c  = getTransformedCoefficientsU1(sc1, x1 - x0, y1 - y0);
    Point6d u2c  = getTransformedCoefficientsU2(sc1, x1 - x0, y1 - y0);
    Point6d u2e  = getTransformedCoefficientsU2(sc2, x2 - x0, y2 - y0);
    Point6d u3c  = getTransformedCoefficientsU3(sc1, x1 - x0, y1 - y0);
    Point6d u3e  = getTransformedCoefficientsU3(sc2, x2 - x0, y2 - y0);


/*
    Point6d u0c  = getTransformedCoefficientsU0(sc1, x1, y1);
    Point6d u1c  = getTransformedCoefficientsU1(sc1, x1, y1);
    Point6d u2c  = getTransformedCoefficientsU2(sc1, x1, y1);
    Point6d u2e  = getTransformedCoefficientsU2(sc2, x2, y2);
    Point6d u3c  = getTransformedCoefficientsU3(sc1, x1, y1);
    Point6d u3e  = getTransformedCoefficientsU3(sc2, x2, y2);
*/

    Point6d u2ec = u2e.diff(u2c);
    Point6d u3ec = u3e.diff(u3c);
    Point6d vc0  = u0c.scale(2 * sqr(dx)).diff(u2ec.scale(dx));
    Point6d vc1  = u1c.scale(2 * sqr(dy)).diff(u3ec.scale(dy));
    Point6d vc   = vc0.diff(vc1);

    if (isSelectedPixel(i, j)) {
         std::cout << "=============" << std::endl;
        // std::cout << " sc=" << sc  << std::endl;
         std::cout << "sc1=" << sc1 << std::endl;
         std::cout << "sc2=" << sc2 << std::endl;
         std::cout << "=============" << std::endl;
    }

    gsl_matrix_set (A, k, 0, vc.x0);
    gsl_matrix_set (A, k, 1, vc.x1);
    gsl_matrix_set (A, k, 2, vc.x2);
    gsl_matrix_set (A, k, 3, vc.x3);
    gsl_matrix_set (A, k, 4, vc.x4);
    gsl_matrix_set (A, k, 5, vc.x5);
    gsl_vector_set (b, k,   -vc.x5);
    return true;
}

static bool fillRowGsl7(int k, Point6dMatrix& scMatrix, int i, int j, int i1, int j1, int i2, int j2, int rows, int cols, int dim, gsl_matrix * A,  gsl_vector* b) {

    //if (!scMatrix[i][j]) { return false; }
    if (!scMatrix[i1][j1] || !scMatrix[i2][j2]) { return false; }

    //Point6d sc1  = scMatrix[i][j].value();
    //Point6d sc2  = scMatrix[i][j].value();

    Point6d sc1  = scMatrix[i1][j1].value();
    Point6d sc2  = scMatrix[i2][j2].value();

    double x0    = scaleX(j, cols, dim);
    double y0    = scaleY(i, rows, dim);

    double x1    = scaleX(j1, cols, dim);
    double y1    = scaleY(i1, rows, dim);

    double x2    = scaleX(j2, cols, dim);
    double y2    = scaleY(i2, rows, dim);

    //Point6d sc1  = getTranslatedSurfaceCoefficients(scMatrix[i][j].value(), x0, y0, x1, y1);
    //Point6d sc2  = getTranslatedSurfaceCoefficients(scMatrix[i][j].value(), x0, y0, x2, y2);

    //Point6d sc1  = sc;
    //Point6d sc2  = sc;

    
    double dx = x2 - x1;
    double dy = y2 - y1;

    if (isSelectedPixel(i, j)) {
         std::cout << "=============" << std::endl;
        // std::cout << " sc=" << sc  << std::endl;
         std::cout << "sc1=" << sc1 << std::endl;
         std::cout << "sc2=" << sc2 << std::endl;
    }

    Point6d u0c  = getTransformedCoefficientsU0(sc1, x1, y1);
    Point6d u0e  = getTransformedCoefficientsU0(sc2, x2, y2);
    Point6d vc0 = u0c.diff(u0e);

    Point6d u1c  = getTransformedCoefficientsU1(sc1, x1, y1);
    Point6d u1e  = getTransformedCoefficientsU1(sc2, x2, y2);
    Point6d vc1 = u1c.diff(u1e);

    if (isSelectedPixel(i, j)) {
        // std::cout << " sc=" << sc  << std::endl;
        std::cout << "u0c=" << u0c << std::endl;
        std::cout << "u0e=" << u0e << std::endl;
        std::cout << "u1c=" << u1c << std::endl;
        std::cout << "u1e=" << u1e << std::endl;
        std::cout << "vc0=" << vc0 << std::endl;
        std::cout << "vc1=" << vc1 << std::endl;
        std::cout << "=============" << std::endl;
    }

/*
    gsl_matrix_set (A, k, 0, vc1.x1);
    gsl_matrix_set (A, k, 1, vc1.x2);
    gsl_matrix_set (A, k, 2, vc1.x4);
    gsl_vector_set (b, k,   -vc1.x5);
*/
/*
    gsl_matrix_set (A, k, 0, vc0.x0);
    gsl_matrix_set (A, k, 1, vc0.x1);
    gsl_matrix_set (A, k, 2, vc0.x2);
    gsl_matrix_set (A, k, 3, vc0.x3);
    gsl_matrix_set (A, k, 4, vc0.x4);
    gsl_matrix_set (A, k, 5, vc0.x5);
    gsl_vector_set (b, k,   -vc0.x5);
*/
/*
    gsl_matrix_set (A, k, 0, vc1.x0);
    gsl_matrix_set (A, k, 1, vc1.x1);
    gsl_matrix_set (A, k, 2, vc1.x2);
    gsl_matrix_set (A, k, 3, vc1.x3);
    gsl_matrix_set (A, k, 4, vc1.x4);
    gsl_matrix_set (A, k, 5, vc1.x5);
    gsl_vector_set (b, k,   -vc1.x5);
*/

    gsl_matrix_set (A, 2 * k, 0, vc0.x0);
    gsl_matrix_set (A, 2 * k, 1, vc0.x1);
    gsl_matrix_set (A, 2 * k, 2, vc0.x2);
    gsl_matrix_set (A, 2 * k, 3, vc0.x3);
    gsl_matrix_set (A, 2 * k, 4, vc0.x4);
   // gsl_matrix_set (A, 2 * k, 5, vc0.x5);
   // gsl_vector_set (b, 2 * k,   -vc0.x5);

    gsl_matrix_set (A, 2 * k + 1, 0, vc1.x0);
    gsl_matrix_set (A, 2 * k + 1, 1, vc1.x1);
    gsl_matrix_set (A, 2 * k + 1, 2, vc1.x2);
    gsl_matrix_set (A, 2 * k + 1, 3, vc1.x3);
    gsl_matrix_set (A, 2 * k + 1, 4, vc1.x4);

  //  gsl_matrix_set (A, 2 * k + 1, 5, vc1.x5);
  //  gsl_vector_set (b, 2 * k + 1,   -vc1.x5);

   return true;
}

static bool fillRowGsl8(int k, Point6dMatrix& scMatrix, int i0, int j0, int i, int j, int rows, int cols, int dim, gsl_matrix * A,  gsl_vector* b) {

    if (!scMatrix[i][j]) { 
        return {}; 
    }

    double x0    = scaleX(j0, cols, dim);
    double y0    = scaleY(i0, rows, dim);

    double x    = scaleX(j, cols, dim);
    double y    = scaleY(i, rows, dim);

    //Point6d sc0  = scMatrix[i0][j0].value();
    //Point6d sc  = getTranslatedSurfaceCoefficients(sc0, x0, y0, x, y);
    Point6d sc  = scMatrix[i][j].value();

    Point6d vc0  = getTransformedCoefficientsU0(sc, x - x0, y - y0);
    Point6d vc1  = getTransformedCoefficientsU1(sc, x - x0, y - y0);

    if (isSelectedPixel(i, j)) {
         std::cout << "=============" << std::endl;
        // std::cout << " sc=" << sc  << std::endl;
         std::cout << "sc=" << sc << std::endl;
         std::cout << "vc0=" << vc0 << std::endl;
         std::cout << "vc1=" << vc1 << std::endl;
         std::cout << "=============" << std::endl;
    }

    gsl_matrix_set (A, k, 0, vc0.x0);
    gsl_matrix_set (A, k, 1, vc0.x1);
    gsl_matrix_set (A, k, 2, vc0.x2);
    gsl_matrix_set (A, k, 3, vc0.x3);
    gsl_matrix_set (A, k, 4, vc0.x4);
    gsl_matrix_set (A, k, 5, vc0.x5);
    gsl_matrix_set (A, k, 6, -1);
    gsl_matrix_set (A, k, 7, 0);
    gsl_vector_set (b, k + 1,   -vc0.x5);

    gsl_matrix_set (A, k + 1, 0, vc1.x0);
    gsl_matrix_set (A, k + 1, 1, vc1.x1);
    gsl_matrix_set (A, k + 1, 2, vc1.x2);
    gsl_matrix_set (A, k + 1, 3, vc1.x3);
    gsl_matrix_set (A, k + 1, 4, vc1.x4);
    gsl_matrix_set (A, k + 1, 5, vc1.x5);
    gsl_matrix_set (A, k + 1, 6, 0);
    gsl_matrix_set (A, k + 1, 7, -1);
    gsl_vector_set (b, k + 1,   -vc1.x5);
    return true;
}


static boost::optional<Point6d> calculate_normal_coefficients(Point6dMatrix& scMatrix, int i, int j, int rows, int cols, int dim, int w, int lastRow, Point3d & lightVector) {

    if (i < w || i >= (rows - w) || j < w || j >= (cols - w)) {
        return {};
    }
    vector<std::pair<int, int>> pixels;

    pixels.push_back(std::pair<int, int>(j    , i - w));
    pixels.push_back(std::pair<int, int>(j    , i + w));
    pixels.push_back(std::pair<int, int>(j - w,     i));
    pixels.push_back(std::pair<int, int>(j + w,     i));

    pixels.push_back(std::pair<int, int>(j - w, i - w));
    pixels.push_back(std::pair<int, int>(j - w, i + w));

    pixels.push_back(std::pair<int, int>(j + w, i - w));
    pixels.push_back(std::pair<int, int>(j + w, i + w));

/*
    int w_r2 = w * 0.707;
    int w2 = w / 2;
    int w3 = w / 3;

    pixels.push_back(std::pair<int, int>(j - w_r2, i - w_r2));
    pixels.push_back(std::pair<int, int>(j - w_r2, i + w_r2));
    pixels.push_back(std::pair<int, int>(j + w_r2, i + w_r2));
    pixels.push_back(std::pair<int, int>(j + w_r2, i - w_r2));

    pixels.push_back(std::pair<int, int>(j - w, i - w2));
    pixels.push_back(std::pair<int, int>(j - w, i + w2));
    pixels.push_back(std::pair<int, int>(j + w, i + w2));
    pixels.push_back(std::pair<int, int>(j + w, i - w2));

    pixels.push_back(std::pair<int, int>(j - w, i - w3));
    pixels.push_back(std::pair<int, int>(j - w, i + w3));
    pixels.push_back(std::pair<int, int>(j + w, i + w3));
    pixels.push_back(std::pair<int, int>(j + w, i - w3));

    pixels.push_back(std::pair<int, int>(j + w2, i - w));
    pixels.push_back(std::pair<int, int>(j - w2, i - w));
    pixels.push_back(std::pair<int, int>(j + w2, i + w));
    pixels.push_back(std::pair<int, int>(j - w2, i + w));

    pixels.push_back(std::pair<int, int>(j + w3, i - w));
    pixels.push_back(std::pair<int, int>(j - w3, i - w));
    pixels.push_back(std::pair<int, int>(j + w3, i + w));
    pixels.push_back(std::pair<int, int>(j - w3, i + w));
*/
    gsl_matrix * A = gsl_matrix_alloc(pixels.size(), 6);
    gsl_vector * b = gsl_vector_alloc(pixels.size());

    for (int k = 0; k < pixels.size(); k++) {
        if (!fillRowGsl(k, scMatrix, i, j, pixels[k], rows, cols, dim, lightVector, A, b)) {
            return {};
        }
    }

    gsl_matrix * V    = gsl_matrix_alloc (6, 6);
    gsl_vector * S    = gsl_vector_alloc(6);
    gsl_vector * work = gsl_vector_alloc(6);
    gsl_linalg_SV_decomp (A, V, S, work);
    gsl_vector* x     = gsl_vector_alloc(6);
    gsl_linalg_SV_solve (A, V, S, b, x);

    if (isSelectedPixel(i, j)) {
        std::cout << "A=" << std::endl;
        print_matrix(A);
        std::cout << "det(A)=" <<  gsl_linalg_LU_det(A, 1) << std::endl;
        std::cout << "S=" << std::endl;
        for (int k = 0; k < 6; k++) {
            std::cout << gsl_vector_get(S, k) << std::endl;        
        }        
        std::cout << "V=" << std::endl;
        print_matrix(V);
        std::cout << "b=" << std::endl;
        for (int k = 0; k < 6; k++) {
            std::cout << gsl_vector_get(b, k) << std::endl;
        
        }        
        std::cout << "x=" << std::endl;
        for (int k = 0; k < 6; k++) {
            std::cout << gsl_vector_get(x, k) << std::endl;
        
        }        
    }

/*
    double u0 = gsl_vector_get(x, 0);
    double u1 = gsl_vector_get(x, 1);
    double u2 = gsl_vector_get(x, 2);
    double u3 = gsl_vector_get(x, 3);
    double u4 = gsl_vector_get(x, 4);
    double u5 = gsl_vector_get(x, 5) + 1;
*/

    int lastCol = 5;
    double u0 = gsl_matrix_get(V, 0, lastCol);
    double u1 = gsl_matrix_get(V, 1, lastCol);
    double u2 = gsl_matrix_get(V, 2, lastCol);
    double u3 = gsl_matrix_get(V, 3, lastCol);
    double u4 = gsl_matrix_get(V, 4, lastCol);
    double u5 = gsl_matrix_get(V, 5, lastCol);

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
    if (isSelectedPixel(i, j)) {
        std::cout << "calculate_normal_coefficients" << std::endl;
        std::cout << "i=" << i << " j=" << j << std::endl;
        std::cout << "u     =" << u << std::endl;
        std::cout << "u n=" << u.normalize2() << std::endl;
    }

/*
    u = u.normalize();
    if (isSelectedPixel(i, j)) {
        std::cout << "calculate_normal_coefficients i=" << i << " j=" << j << " u normalized=" << u << std::endl;
    }
*/
    return u;    
}

static boost::optional<Point6d> calculate_normal_coefficients1(Point6dMatrix& scMatrix, int i, int j, int rows, int cols, int dim, int w, int lastRow, Point3d & lightVector) {

    if (i < w || i >= (rows - w) || j < w || j >= (cols - w) || !scMatrix[i][j]) {
        return {};
    }
    vector<std::pair<int, int>> pixels;

    pixels.push_back(std::pair<int, int>(j    , i - w));
    pixels.push_back(std::pair<int, int>(j    , i + w));
    pixels.push_back(std::pair<int, int>(j - w,     i));
    pixels.push_back(std::pair<int, int>(j + w,     i));

    pixels.push_back(std::pair<int, int>(j - w, i - w));
    pixels.push_back(std::pair<int, int>(j - w, i + w));
    pixels.push_back(std::pair<int, int>(j + w, i - w));
    pixels.push_back(std::pair<int, int>(j + w, i + w));

    gsl_matrix * A = gsl_matrix_alloc(pixels.size(), 5);
    gsl_vector * b = gsl_vector_alloc(pixels.size());

    for (int k = 0; k < pixels.size(); k++) {
        if (!fillRowGsl1(k, scMatrix, i, j, pixels[k], rows, cols, dim, A, b, lightVector)) {
            return {};
        }
    }

    gsl_matrix * V    = gsl_matrix_alloc (5, 5);
    gsl_vector * S    = gsl_vector_alloc(5);
    gsl_vector * work = gsl_vector_alloc(5);
    gsl_linalg_SV_decomp (A, V, S, work);
    gsl_vector* x     = gsl_vector_alloc(5);
    gsl_linalg_SV_solve (A, V, S, b, x);

    if (isSelectedPixel(i, j)) {
        std::cout << "A=" << std::endl;
        print_matrix(A);
        std::cout << "det(A)=" <<  gsl_linalg_LU_det(A, 1) << std::endl;
        std::cout << "S=" << std::endl;
        for (int k = 0; k < 5; k++) {
            std::cout << gsl_vector_get(S, k) << std::endl;        
        }        
        std::cout << "V=" << std::endl;
        print_matrix(V);
        std::cout << "b=" << std::endl;
        for (int k = 0; k < 5; k++) {
            std::cout << gsl_vector_get(b, k) << std::endl;
        
        }        
        std::cout << "x=" << std::endl;
        for (int k = 0; k < 5; k++) {
            std::cout << gsl_vector_get(x, k) << std::endl;
        
        }        
    }

    int lastCol = 4;

    double X    = scaleX(j, cols, dim);
    double Y    = scaleY(i, rows, dim);

    Point3d zero(0.0, 0.0, 0.0);
    Point6d vc  = getTransformedCoefficients(scMatrix[i][j].value(), X , Y, zero);

    double u0 = gsl_matrix_get(V, 0, lastCol);
    double u1 = gsl_matrix_get(V, 1, lastCol);
    double u2 = gsl_matrix_get(V, 2, lastCol);
    double u3 = gsl_matrix_get(V, 3, lastCol);
    double u4 = gsl_matrix_get(V, 4, lastCol);
    double u5 = (vc.x0 * u0 + vc.x1 * u1 + vc.x2 * u2 + vc.x3 * u3 + vc.x4 * u5) / vc.x5;

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
    if (isSelectedPixel(i, j)) {
        std::cout << "calculate_normal_coefficients i=" << i << " j=" << j << " u=" << u << std::endl;
    }

    return u;    
}

static boost::optional<Point6d> calculate_normal_coefficients2(Point6dMatrix& scMatrix, int i, int j, int rows, int cols, int dim, int w, int lastRow, Point3d & lightVector) {

    if (i < w || i >= (rows - w) || j < w || j >= (cols - w) || !scMatrix[i][j]) {
        return {};
    }

    gsl_matrix * A = gsl_matrix_alloc(8, 5);
    gsl_vector * b = gsl_vector_alloc(8 );

    int w2 = w / 2;
    int w3 = w / 3;
    int w4 = w / 4;

    if (!fillRowGsl2(0, scMatrix, i, j, i - w, j - w, i + w, j + w, rows, cols, dim, A, b, lightVector)) {
        return {};
    }

    if (!fillRowGsl2(1, scMatrix, i, j, i - w, j + w, i + w, j - w, rows, cols, dim, A, b, lightVector)) {
        return {};
    }

    if (!fillRowGsl2(2, scMatrix, i, j, i - w2, j - w, i + w2, j + w, rows, cols, dim, A, b, lightVector)) {
        return {};
    }

    if (!fillRowGsl2(3, scMatrix, i, j, i - w, j + w2, i + w, j - w2, rows, cols, dim, A, b, lightVector)) {
        return {};
    }

    if (!fillRowGsl2(4, scMatrix, i, j, i - w3, j - w, i + w3, j + w, rows, cols, dim, A, b, lightVector)) {
        return {};
    }

    if (!fillRowGsl2(5, scMatrix, i, j, i - w, j + w3, i + w, j - w3, rows, cols, dim, A, b, lightVector)) {
        return {};
    }

    if (!fillRowGsl2(6, scMatrix, i, j, i - w3, j - w, i + w3, j + w, rows, cols, dim, A, b, lightVector)) {
        return {};
    }

    if (!fillRowGsl2(7, scMatrix, i, j, i - w, j + w3, i + w, j - w3, rows, cols, dim, A, b, lightVector)) {
        return {};
    }


    gsl_matrix * V    = gsl_matrix_alloc (5, 5);
    gsl_vector * S    = gsl_vector_alloc(5);
    gsl_vector * work = gsl_vector_alloc(5);
    gsl_linalg_SV_decomp (A, V, S, work);
    gsl_vector* x     = gsl_vector_alloc(5);
    gsl_linalg_SV_solve (A, V, S, b, x);

    if (isSelectedPixel(i, j)) {
        std::cout << "A=" << std::endl;
        print_matrix(A);
        std::cout << "det(A)=" <<  gsl_linalg_LU_det(A, 1) << std::endl;
        std::cout << "S=" << std::endl;
        for (int k = 0; k < 5; k++) {
            std::cout << gsl_vector_get(S, k) << std::endl;        
        }        
        std::cout << "V=" << std::endl;
        print_matrix(V);
        std::cout << "b=" << std::endl;
        for (int k = 0; k < 5; k++) {
            std::cout << gsl_vector_get(b, k) << std::endl;
        
        }        
        std::cout << "x=" << std::endl;
        for (int k = 0; k < 5; k++) {
            std::cout << gsl_vector_get(x, k) << std::endl;
        
        }        
    }

    int lastCol = 4;

    double X    = scaleX(j, cols, dim);
    double Y    = scaleY(i, rows, dim);

    Point3d zero(0.0, 0.0, 0.0);
    Point6d vc  = getTransformedCoefficients(scMatrix[i][j].value(), X , Y, zero);

    double u0 = gsl_matrix_get(V, 0, lastCol);
    double u1 = gsl_matrix_get(V, 1, lastCol);
    double u2 = gsl_matrix_get(V, 2, lastCol);
    double u3 = gsl_matrix_get(V, 3, lastCol);
    double u4 = gsl_matrix_get(V, 4, lastCol);
    double u5 = (vc.x0 * u0 + vc.x1 * u1 + vc.x2 * u2 + vc.x3 * u3 + vc.x4 * u5) / vc.x5;

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
    if (isSelectedPixel(i, j)) {
        std::cout << "calculate_normal_coefficients i=" << i << " j=" << j << " u=" << u << std::endl;
    }

    return u;    
}

static boost::optional<Point6d> calculate_normal_coefficients4(Point6dMatrix& scMatrix, int i, int j, int rows, int cols, int dim, int w, int lastRow, Point3d & lightVector) {

    if (i < w || i >= (rows - w) || j < w || j >= (cols - w)) {
        return {};
    }
    vector<std::pair<int, int>> pixels;

    int w2 = w / 2;
    int w3 = w / 3;
    int w4 = w / 4;
    int w5 = w / 5;

    pixels.push_back(std::pair<int, int>(j - w , i + w ));
    pixels.push_back(std::pair<int, int>(j - w , i - w ));

    pixels.push_back(std::pair<int, int>(j - w , i + w2));
    pixels.push_back(std::pair<int, int>(j - w , i - w2));

    pixels.push_back(std::pair<int, int>(j - w2, i - w ));
    pixels.push_back(std::pair<int, int>(j + w2, i - w ));

    pixels.push_back(std::pair<int, int>(j - w2, i + w ));
    pixels.push_back(std::pair<int, int>(j + w2, i + w ));

/*
    pixels.push_back(std::pair<int, int>(j - w , i + w3));
    pixels.push_back(std::pair<int, int>(j - w , i - w3));

    pixels.push_back(std::pair<int, int>(j - w3, i - w ));
    pixels.push_back(std::pair<int, int>(j + w3, i - w ));

    pixels.push_back(std::pair<int, int>(j - w3, i + w ));
    pixels.push_back(std::pair<int, int>(j + w3, i + w ));

    pixels.push_back(std::pair<int, int>(j - w , i + w4));
    pixels.push_back(std::pair<int, int>(j - w , i - w4));

    pixels.push_back(std::pair<int, int>(j - w4, i - w ));
    pixels.push_back(std::pair<int, int>(j + w4, i - w ));

    pixels.push_back(std::pair<int, int>(j - w5, i + w ));
    pixels.push_back(std::pair<int, int>(j + w5, i + w ));

    pixels.push_back(std::pair<int, int>(j - w , i + w5));
    pixels.push_back(std::pair<int, int>(j - w , i - w5));

    pixels.push_back(std::pair<int, int>(j - w5, i - w ));
    pixels.push_back(std::pair<int, int>(j + w5, i - w ));

    pixels.push_back(std::pair<int, int>(j - w5, i + w ));
    pixels.push_back(std::pair<int, int>(j + w5, i + w ));
*/
/*
    pixels.push_back(std::pair<int, int>(j - w, i));
    pixels.push_back(std::pair<int, int>(j + w, i));
    pixels.push_back(std::pair<int, int>(j    , i - w));
    pixels.push_back(std::pair<int, int>(j    , i + w));

    pixels.push_back(std::pair<int, int>(j - w, i - w));
    pixels.push_back(std::pair<int, int>(j - w, i + w));
    pixels.push_back(std::pair<int, int>(j + w, i - w));
    pixels.push_back(std::pair<int, int>(j + w, i + w));

    int w2 = w / 2;
    pixels.push_back(std::pair<int, int>(j - w2, i));
    pixels.push_back(std::pair<int, int>(j + w2, i));
    pixels.push_back(std::pair<int, int>(j    , i - w2));
    pixels.push_back(std::pair<int, int>(j    , i + w2));

    pixels.push_back(std::pair<int, int>(j - w2, i - w2));
    pixels.push_back(std::pair<int, int>(j - w2, i + w2));
    pixels.push_back(std::pair<int, int>(j + w2, i - w2));
    pixels.push_back(std::pair<int, int>(j + w2, i + w2));

    int w3 = w / 3;
    pixels.push_back(std::pair<int, int>(j - w3, i));
    pixels.push_back(std::pair<int, int>(j + w3, i));
    pixels.push_back(std::pair<int, int>(j    , i - w3));
    pixels.push_back(std::pair<int, int>(j    , i + w3));

    pixels.push_back(std::pair<int, int>(j - w3, i - w3));
    pixels.push_back(std::pair<int, int>(j - w3, i + w3));
    pixels.push_back(std::pair<int, int>(j + w3, i - w3));
    pixels.push_back(std::pair<int, int>(j + w3, i + w3));
*/
   // pixels.push_back(std::pair<int, int>(j - w, i - w3));
   // pixels.push_back(std::pair<int, int>(j - w, i - w2));
   // pixels.push_back(std::pair<int, int>(j - w3, i - w));
   // pixels.push_back(std::pair<int, int>(j - w2, i - w));
   // pixels.push_back(std::pair<int, int>(j + w3, i - w));
   // pixels.push_back(std::pair<int, int>(j + w2, i - w));
   // pixels.push_back(std::pair<int, int>(j + w, i - w3));
   //  pixels.push_back(std::pair<int, int>(j + w, i - w2));

/*
    pixels.push_back(std::pair<int, int>(j - w2, i - w));
    pixels.push_back(std::pair<int, int>(j - w3, i - w));
    pixels.push_back(std::pair<int, int>(j + w2, i - w));
    pixels.push_back(std::pair<int, int>(j + w3, i - w));
    pixels.push_back(std::pair<int, int>(j + w , i - w2));
    pixels.push_back(std::pair<int, int>(j + w , i - w3));
    pixels.push_back(std::pair<int, int>(j + w , i + w2));
    pixels.push_back(std::pair<int, int>(j + w , i + w3));
    pixels.push_back(std::pair<int, int>(j + w2, i + w));
    pixels.push_back(std::pair<int, int>(j + w3, i + w));
    pixels.push_back(std::pair<int, int>(j - w2, i + w));
    pixels.push_back(std::pair<int, int>(j - w3, i + w));
    pixels.push_back(std::pair<int, int>(j - w , i + w2));
    pixels.push_back(std::pair<int, int>(j - w , i + w3));
*/
    int cnt_vars = 6;
    gsl_matrix * A = gsl_matrix_alloc(pixels.size(), cnt_vars);
    gsl_vector * b = gsl_vector_alloc(pixels.size());

    for (int k = 0; k < pixels.size(); k++) {
        if (!fillRowGsl6(k, scMatrix, i, j, pixels[k], rows, cols, dim, A, b)) {
            return {};
        }
    }

    if (isSelectedPixel(i, j)) {
        std::cout << "A before=" << std::endl;
        print_matrix(A);
        std::cout << "det(A before)=" <<  gsl_linalg_LU_det(A, 1) << std::endl;
    }
    gsl_matrix * V    = gsl_matrix_alloc(cnt_vars, cnt_vars);
    gsl_vector * S    = gsl_vector_alloc(cnt_vars);
    gsl_vector * work = gsl_vector_alloc(cnt_vars);
    gsl_linalg_SV_decomp (A, V, S, work);
    gsl_vector* x     = gsl_vector_alloc(cnt_vars);
    gsl_linalg_SV_solve (A, V, S, b, x);

    if (isSelectedPixel(i, j)) {
        std::cout << "A=" << std::endl;
        print_matrix(A);
        std::cout << "det(A)=" <<  gsl_linalg_LU_det(A, 1) << std::endl;
        std::cout << "S=" << std::endl;
        for (int k = 0; k < cnt_vars; k++) {
            std::cout << gsl_vector_get(S, k) << std::endl;        
        }        
        std::cout << "V=" << std::endl;
        print_matrix(V);
        std::cout << "b=" << std::endl;
        for (int k = 0; k < cnt_vars; k++) {
            std::cout << gsl_vector_get(b, k) << std::endl;
        
        }        
        std::cout << "x=" << std::endl;
        for (int k = 0; k < cnt_vars; k++) {
            std::cout << gsl_vector_get(x, k) << std::endl;
        
        }        
    }

    int lastCol = cnt_vars - 1;
    double u0 = gsl_matrix_get(V, 0, lastCol);
    double u1 = gsl_matrix_get(V, 1, lastCol);
    double u2 = gsl_matrix_get(V, 2, lastCol);
    double u3 = gsl_matrix_get(V, 3, lastCol);
    double u4 = gsl_matrix_get(V, 4, lastCol);
    double u5 = gsl_matrix_get(V, 5, lastCol);

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
    if (isSelectedPixel(i, j)) {
        std::cout << "calculate_normal_coefficients i=" << i << " j=" << j << " u=" << u << std::endl;
    }

/*
    u = u.normalize();
    if (isSelectedPixel(i, j)) {
        std::cout << "calculate_normal_coefficients i=" << i << " j=" << j << " u normalized=" << u << std::endl;
    }
*/

    gsl_matrix_free(A);
    gsl_matrix_free(V);
    gsl_vector_free(S);
    gsl_vector_free(b);
    gsl_vector_free(work);
    gsl_vector_free(x);
    return u;    
}

static boost::optional<Point6d> calculate_normal_coefficients5(Point6dMatrix& scMatrix, int i, int j, int rows, int cols, int dim, int w, int lastRow, Point3d & lightVector) {

    if (i < w || i >= (rows - w) || j < w || j >= (cols - w)) {
        return {};
    }
    vector<std::pair<int, int>> pixels;

    int w2 = w / 2;
    int w3 = w / 3;
    int w4 = w / 4;

    int cnt_vars = 5;

/*
    for (int k = -w; k <= 0; k++) {
        for (int m = -w; m < w; m++) {
            if (k == 0 && m == 0) {
                continue;
            }
            pixels.push_back(std::pair<int, int>(m, k));       
        }
    }
*/
    //int sz = pixels.size();
    //int sz = 16;
    int sz = 8;

//    std::cout << "sz=" << sz << std::endl;

    gsl_matrix * A = gsl_matrix_alloc(sz, cnt_vars);
    gsl_vector * b = gsl_vector_alloc(sz);
    int row = 0;

/*
    for (int k = 0; k < sz; k++) {
        std::pair<int, int> p = pixels[k];
        int f = p.first;
        int s = p.second;
        if (!fillRowGsl7(row++, scMatrix, i, j, i - s, j - f, i + s, j + f, rows, cols, dim, A, b)) {
            return {};
        }
    }
*/
/*
    if (!fillRowGsl7(row++, scMatrix, i, j, i - w , j - w , i - w , j + w , rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i - w2, j - w , i - w2, j + w , rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i     , j - w , i     , j + w , rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i + w2, j - w , i + w2, j + w , rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i + w , j - w , i + w , j + w , rows, cols, dim, A, b)) return {};
*/

/*
    if (!fillRowGsl7(row++, scMatrix, i, j, i - w , j - w , i - w, j + w , rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i - w2, j - w , i - w2, j + w , rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i - w3, j - w , i - w3, j + w , rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i     , j - w , i     , j + w , rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i + w2, j - w , i + w2, j + w , rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i + w3, j - w , i + w3, j + w , rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i + w , j - w , i + w, j + w , rows, cols, dim, A, b)) return {};
*/

/*
    if (!fillRowGsl7(row++, scMatrix, i, j, i - w , j - w  , i + w , j - w , rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i - w , j - w2 , i + w , j - w2, rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i - w , j - w3 , i + w , j - w3, rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i - w , j      , i + w , j     , rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i - w , j + w  , i + w , j + w , rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i - w , j + w2 , i + w , j + w2, rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i - w , j + w3 , i + w , j + w3, rows, cols, dim, A, b)) return {};
*/

    if (!fillRowGsl7(row++, scMatrix, i, j, i , j - w , i , j + w , rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i , j - w2 , i, j + w2 , rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i , j - w3 , i , j + w3 , rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i , j - w4 , i , j + w4 , rows, cols, dim, A, b)) return {};

/*
    if (!fillRowGsl7(row++, scMatrix, i, j, i - w  , j - w , i + w , j + w , rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i + w  , j - w , i - w , j + w , rows, cols, dim, A, b)) return {}; 

    if (!fillRowGsl7(row++, scMatrix, i, j, i - w2  , j - w2 , i + w2 , j + w2 , rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i + w2  , j - w2 , i - w2 , j + w2 , rows, cols, dim, A, b)) return {}; 

    if (!fillRowGsl7(row++, scMatrix, i, j, i - w3  , j - w3 , i + w3 , j + w3 , rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i + w3  , j - w3 , i - w3 , j + w3 , rows, cols, dim, A, b)) return {}; 

    if (!fillRowGsl7(row++, scMatrix, i, j, i - w4  , j - w4 , i + w4 , j + w4 , rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i + w4  , j - w4 , i - w4 , j + w4 , rows, cols, dim, A, b)) return {}; 
*/
/*
    if (!fillRowGsl7(row++, scMatrix, i, j, i - w2 , j - w2, i + w2, j + w2, rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i + w2 , j - w2, i - w2, j + w2, rows, cols, dim, A, b)) return {}; 

    if (!fillRowGsl7(row++, scMatrix, i, j, i - w3 , j - w3, i + w3, j + w3, rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i + w3 , j - w3, i - w3, j + w3, rows, cols, dim, A, b)) return {};

    if (!fillRowGsl7(row++, scMatrix, i, j, i - w4 , j - w4, i + w4, j + w4, rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i + w4 , j - w4, i - w4, j + w4, rows, cols, dim, A, b)) return {};
*/
/*
    if (!fillRowGsl7(row++, scMatrix, i, j, i - w , j - w2, i + w , j - w2, rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i - w , j - w3, i + w , j - w3, rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i - w , j     , i + w , j     , rows, cols, dim, A, b)) return {};

    if (!fillRowGsl7(row++, scMatrix, i, j, i - w , j + w2, i + w , j + w2, rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i - w , j + w3, i + w , j + w3, rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i - w , j + w , i + w , j + w , rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i - w , j - w , i + w , j + w , rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i + w , j - w , i - w , j + w , rows, cols, dim, A, b)) return {};

    if (!fillRowGsl7(row++, scMatrix, i, j, i - w , j - w2, i + w , j + w2 , rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i + w , j - w2, i - w , j + w2 , rows, cols, dim, A, b)) return {};

    if (!fillRowGsl7(row++, scMatrix, i, j, i - w2, j - w , i + w2, j + w , rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i + w2, j - w , i - w2 , j + w , rows, cols, dim, A, b)) return {};

    if (!fillRowGsl7(row++, scMatrix, i, j, i - w , j - w3, i + w , j + w3 , rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i + w , j - w3, i - w , j + w3 , rows, cols, dim, A, b)) return {};

    if (!fillRowGsl7(row++, scMatrix, i, j, i - w3, j - w , i + w3, j + w , rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i + w3, j - w , i - w3, j + w , rows, cols, dim, A, b)) return {};
*/

/*
    if (!fillRowGsl7(row++, scMatrix, i, j, i - w3 , j - w3 , i + w3 , j + w3 , rows, cols, dim, A, b)) return {};
    if (!fillRowGsl7(row++, scMatrix, i, j, i + w3 , j - w3 , i - w3 , j + w3 , rows, cols, dim, A, b)) return {};
*/
    if (isSelectedPixel(i, j)) {
        std::cout << "A before=" << std::endl;
        print_matrix(A);
        std::cout << "det(A before)=" <<  gsl_linalg_LU_det(A, 1) << std::endl;
    }
    gsl_matrix * V    = gsl_matrix_alloc(cnt_vars, cnt_vars);
    gsl_vector * S    = gsl_vector_alloc(cnt_vars);
    gsl_vector * work = gsl_vector_alloc(cnt_vars);
    gsl_linalg_SV_decomp (A, V, S, work);
    gsl_vector* x     = gsl_vector_alloc(cnt_vars);
    gsl_linalg_SV_solve (A, V, S, b, x);

    if (isSelectedPixel(i, j)) {
        std::cout << "A=" << std::endl;
        print_matrix(A);
        std::cout << "det(A)=" <<  gsl_linalg_LU_det(A, 1) << std::endl;
        std::cout << "S=" << std::endl;
        for (int k = 0; k < cnt_vars; k++) {
            std::cout << gsl_vector_get(S, k) << std::endl;        
        }        
        std::cout << "V=" << std::endl;
        print_matrix(V);
        std::cout << "b=" << std::endl;
        for (int k = 0; k < cnt_vars; k++) {
            std::cout << gsl_vector_get(b, k) << std::endl;
        
        }        
        std::cout << "x=" << std::endl;
        for (int k = 0; k < cnt_vars; k++) {
            std::cout << gsl_vector_get(x, k) << std::endl;
        
        }        
    }

    int lastCol = cnt_vars - 1;
    double u0 = gsl_matrix_get(V, 0, lastCol);
    double u1 = gsl_matrix_get(V, 1, lastCol);
    double u2 = gsl_matrix_get(V, 2, lastCol);
    double u3 = gsl_matrix_get(V, 3, lastCol);
    double u4 = gsl_matrix_get(V, 4, lastCol);
    double u5 = 0;//gsl_matrix_get(V, 5, lastCol);

    //double u0 = 0; //gsl_matrix_get(V, 0, lastCol);
    //double u1 = gsl_matrix_get(V, 0, lastCol);
    //double u2 = gsl_matrix_get(V, 1, lastCol);
    //double u3 = 0; //gsl_matrix_get(V, 3, lastCol);
    //double u4 = gsl_matrix_get(V, 2, lastCol);
    //double u5 = 0; //gsl_matrix_get(V, 5, lastCol);

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
    if (isSelectedPixel(i, j)) {
        std::cout << "calculate_normal_coefficients i=" << i << " j=" << j << " u=" << u << std::endl;
    }

/*
    u = u.normalize();
    if (isSelectedPixel(i, j)) {
        std::cout << "calculate_normal_coefficients i=" << i << " j=" << j << " u normalized=" << u << std::endl;
    }
*/

    gsl_matrix_free(A);
    gsl_matrix_free(V);
    gsl_vector_free(S);
    gsl_vector_free(b);
    gsl_vector_free(work);
    gsl_vector_free(x);
    return u;    
}

static boost::optional<Point6d> calculate_normal_coefficients6(Point6dMatrix& scMatrix, int i, int j, int rows, int cols, int dim, int w, int lastRow, Point3d & lightVector) {

    if (i < w || i >= (rows - w) || j < w || j >= (cols - w)) {
        return {};
    }
    vector<std::pair<int, int>> pixels;

    int w2 = w / 2;

    pixels.push_back(std::pair<int, int>(j - w , i + w ));
    pixels.push_back(std::pair<int, int>(j - w , i - w ));

    pixels.push_back(std::pair<int, int>(j - w , i + w2));
    pixels.push_back(std::pair<int, int>(j - w , i - w2));

    pixels.push_back(std::pair<int, int>(j - w2, i - w ));
    pixels.push_back(std::pair<int, int>(j + w2, i - w ));

    pixels.push_back(std::pair<int, int>(j - w2, i + w ));
    pixels.push_back(std::pair<int, int>(j + w2, i + w ));

    int cnt_vars = 8;
    gsl_matrix * A = gsl_matrix_alloc(pixels.size() * 2, cnt_vars);
    gsl_vector * b = gsl_vector_alloc(pixels.size() * 2);

    for (int k = 0; k < pixels.size(); k++) {
        if (!fillRowGsl8(2 * k, scMatrix, i, j, pixels[k].second, pixels[k].first, rows, cols, dim, A, b)) {
            return {};
        }
    }

    if (isSelectedPixel(i, j)) {
        std::cout << "A before=" << std::endl;
        print_matrix(A);
        std::cout << "det(A before)=" <<  gsl_linalg_LU_det(A, 1) << std::endl;
    }
    gsl_matrix * V    = gsl_matrix_alloc(cnt_vars, cnt_vars);
    gsl_vector * S    = gsl_vector_alloc(cnt_vars);
    gsl_vector * work = gsl_vector_alloc(cnt_vars);
    gsl_linalg_SV_decomp (A, V, S, work);
    gsl_vector* x     = gsl_vector_alloc(cnt_vars);
    gsl_linalg_SV_solve (A, V, S, b, x);

    if (isSelectedPixel(i, j)) {
        std::cout << "A=" << std::endl;
        print_matrix(A);
        std::cout << "det(A)=" <<  gsl_linalg_LU_det(A, 1) << std::endl;
        std::cout << "S=" << std::endl;
        for (int k = 0; k < cnt_vars; k++) {
            std::cout << gsl_vector_get(S, k) << std::endl;        
        }        
        std::cout << "V=" << std::endl;
        print_matrix(V);
        std::cout << "b=" << std::endl;
        for (int k = 0; k < cnt_vars; k++) {
            std::cout << gsl_vector_get(b, k) << std::endl;
        
        }        
        std::cout << "x=" << std::endl;
        for (int k = 0; k < cnt_vars; k++) {
            std::cout << gsl_vector_get(x, k) << std::endl;
        
        }        
    }

    int lastCol = cnt_vars - 1;
    double u0 = gsl_matrix_get(V, 0, lastCol);
    double u1 = gsl_matrix_get(V, 1, lastCol);
    double u2 = gsl_matrix_get(V, 2, lastCol);
    double u3 = gsl_matrix_get(V, 3, lastCol);
    double u4 = gsl_matrix_get(V, 4, lastCol);
    double u5 = gsl_matrix_get(V, 5, lastCol);

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
    if (isSelectedPixel(i, j)) {
        std::cout << "calculate_normal_coefficients i=" << i << " j=" << j << " u=" << u << std::endl;
    }

/*
    u = u.normalize();
    if (isSelectedPixel(i, j)) {
        std::cout << "calculate_normal_coefficients i=" << i << " j=" << j << " u normalized=" << u << std::endl;
    }
*/

    gsl_matrix_free(A);
    gsl_matrix_free(V);
    gsl_vector_free(S);
    gsl_vector_free(b);
    gsl_vector_free(work);
    gsl_vector_free(x);
    return u;    
}

static Point6d calc_a3_a4(double a0, double a1, double a2, double u0, double u1, double u2, double u3, double u4, double u5, double detA, int i, int j) {

     double detAInv = 1 / detA;
     double a3 = detAInv * (4 * a1 * u3 - 2 * a2 * u4);
     double a4 = detAInv * (- 2 * a2 * u3 + 4 * a0 * u4);

     // calculate homogeneity factor
     double s_inv = u5 - (u1 * u3 * u3 - u2 * u3 * u4 + u0 * u4 * u4) * 0.25 * detAInv * detAInv;

     if (isSelectedPixel(i, j)) {
         std::cout << "a3=" << a3 << " a4=" << a4 << " s_inv=" << s_inv << std::endl;
     }

     if (s_inv < 0) {
         s_inv = 0;;
     }

     double s_root = sqrt(1 / s_inv);


     // we always assume that a5 == 0
     return Point6d(a0, a1, a2, a3, a4, 0).scale(s_root);
}

static void addSolutions(double u0, double u1, double u2, double u3, double u4, double u5, double discriminant, vector<Point6d> & solutions, int i, int j) {

     if (isSelectedPixel(i, j)) {
         std::cout << "u0=" << u0 << " u1=" << u1 << " u2=" << u2 << " u3=" << u3 << " u4=" << u4 << " u5=" << u5 << std::endl;
     }

     double a2_squared = 0.25 * (u0 + u1 + discriminant) / (1 + pow((u0 - u1) / u2, 2));
     if (isSelectedPixel(i, j)) {
         std::cout << "a2_squared=" << a2_squared << std::endl;
     }
            
     if (a2_squared < 0) {
         return;
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

     double detA = 16 * a0 * a1 - 4 * a2_squared;
     if (isSelectedPixel(i, j)) {
         std::cout << "detA=" << detA << std::endl;
     }
     if (is0(detA)) {
         // detA cannot be zero
         return;
     }

     if (u2 == 0) {
         solutions.push_back(calc_a3_a4( a0,  a1, 0, u0, u1, u2, u3, u4, u5, detA, i, j));
         solutions.push_back(calc_a3_a4( a0, -a1, 0, u0, u1, u2, u3, u4, u5, detA, i, j));
         solutions.push_back(calc_a3_a4(-a0,  a1, 0, u0, u1, u2, u3, u4, u5, detA, i, j));
         solutions.push_back(calc_a3_a4(-a0, -a1, 0, u0, u1, u2, u3, u4, u5, detA, i, j));
     } else {
         solutions.push_back(calc_a3_a4( a0,  a1,  a2, u0, u1, u2, u3, u4, u5, detA, i, j));
         solutions.push_back(calc_a3_a4(-a0, -a1, -a2, u0, u1, u2, u3, u4, u5, detA, i, j));
     }

     if (isSelectedPixel(i, j)) {
         std::cout << solutions.size() << " solution" << (solutions.size() == 1 ? "" : "s") << " at (" << i << ", " << j << ")" << std::endl;
         for (int k = 0; k < solutions.size(); k++) {
            std::cout << "solution=" << solutions[k] << std::endl;
         }
     }
}


static void calculate_surface_coefficients(boost::optional<Point6d> u, vector<Point6d> & solutions, int i, int j) {
    if (!u) {
        return;
    }

    Point6d uu = u.value();
    double u0 = uu.x0;
    double u1 = uu.x1;
    double u2 = uu.x2;
    double u3 = uu.x3;
    double u4 = uu.x4;
    double u5 = uu.x5;

    double u_2_squared = u2 * u2;

    if (isSelectedPixel(i, j)) {
        std::cout << "u_2_squared=" << u_2_squared << " 4 * u0 * u1=" << (4 * u0 * u1) << std::endl;
        std::cout << "u2=" << u2 << " u_2_squared / (4 * u0 * u1)=" << (u_2_squared / (4 * u0 * u1)) << std::endl;
    }
    
    if (is0(u2) || is0(u_2_squared / (4 * u0 * u1))) {
        addSolutions(u0, u1, 0, u3, u4, u5, 0, solutions, i, j);
        return;
    }

    double discriminant = 4 * u0 * u1 - u_2_squared;
    
    if (isSelectedPixel(i, j)) {
        std::cout << "discrimin=" << discriminant << std::endl;
    }
    
    if (discriminant < 0) {
        if (abs(discriminant) < 0.01) {
            discriminant = 0.0;
        } else {
            return;
        }
    }

    if (discriminant == 0) {
        addSolutions(u0, u1, u2, u3, u4, u5, 0, solutions, i, j);
    } else {
        double d_root = sqrt(discriminant);
        addSolutions(u0, u1, u2, u3, u4, u5, -d_root, solutions, i, j);
        addSolutions(u0, u1, u2, u3, u4, u5, +d_root, solutions, i, j);
    }
}

static Point3d calc_light_from_s_coeffs(Point6d z_coeff, Point6d s_coeff, double x, double y, int i, int j) {
   // calculate I, Ix and Iy from related J values

//    double I  = 1.94029;
//    double Ix = 0;
//    double Iy = 0.913075;

    double I  = sqrt(s_coeff.x0);
    double Ix = 0.5 * s_coeff.x1 / I;
    double Iy = 0.5 * s_coeff.x2 / I;

    // fetch the surface coefficients
    double a0 = z_coeff.x0;
    double a1 = z_coeff.x1;
    double a2 = z_coeff.x2;
    double a3 = z_coeff.x3;
    double a4 = z_coeff.x4;

    // calculate p & q
    //double p = a3;
    //double q = a4;

    double p = 2 * a0 * x + a2 * y + a3;
    double q = 2 * a1 * y + a2 * x + a4;

    if (isSelectedPixel(i, j)) {
        std::cout << "I=" << I << " Ix=" << Ix << " Iy=" << Iy << " p=" << p << " q=" << q << std::endl;
    }

    // calculate length of surface normal
    double normn = sqrt(sqr(p) + sqr(q) + 1);

    // calculate magnitude hessian
    double detA = 4 * a0 * a1 - a2 * a2;

    // l1 & l2 by equation 130
    double l1 = -normn / detA * (2 * a1 * Ix - a2 * Iy     ) - I * p / normn;
    double l2 = -normn / detA * (   -a2 * Ix + 2  * a0 * Iy) - I * q / normn;

    // l3 by equation 131
    double l3 = I * normn + l1 * p + l2 * q;

    double norml = sqrt(sqr(l1) + sqr(l2) + sqr(l3));
    //if (norml > 1.2 || norml < .9) {
    //    return {};
    //}
    return Point3d(l1 / norml, l2 / norml, l3 / norml);
}

static void calculate_light_from_coefficients(boost::optional<Point6d> u, boost::optional<Point6d> s, vector<Point3d> & light_vectors, double x, double y, int i, int j) {

   vector<Point6d> z_coeffs;
   calculate_surface_coefficients(u, z_coeffs, i, j);
   for (int k = 0; k < z_coeffs.size(); k++) {
       light_vectors.push_back(calc_light_from_s_coeffs(z_coeffs[k], s.value(), x, y, i, j));
   }
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
    "{l1                          | 0      | x component of light vector [0..1)}"
    "{l2                          | 0      | y component of light vector [0..1)}"
    "{l3                          | 1      | y component of light vector (0..1]}"
    "{n_coeffs              count | 10     | number of bspline coefficients}"
    "{w_coeffs              count | 6      | window size for calculation normal coefficients}"
    "{w_smooth              count | 7      | filter window size for smoothing}"
    "{generated_derivatives count | 0      | 0 if derivatives are generated 1 if they are computed from input data }"
    "{sc                          |        | path to input sc file }"
    "{si                    count | 0      | the i-index of the sample point }"
    "{sj                    count | 0      | the j-index of the sample point }"
    "{li                    count | 0      | the i-index of the image window for computing the light vector }"
    "{lj                    count | 0      | the j-index of the image window for computing the light vector }"
    "{lw                    count | 0      | the window size of the image window for computing the light vector }"
    "{di                    count | 0      | the i-index of the debug pixel }"
    "{dj                    count | 0      | the j-index of the debug pixel }"
    "{dw                    count | 0      | the window size of debug pixel window }"
    "{test                  count | 0      | the test number to use for generated derivates }"
    "{lastRow               count | 4      | the row in SVD result to pick for the U vector }"
};

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

    double l1 = parser.get<double>("l1");
    std::cout << "l1:" << l1 << std::endl; 
    double l2 = parser.get<double>("l2");
    std::cout << "l2:" << l2 << std::endl; 
    double l3 = parser.get<double>("l3");
    std::cout << "l3:" << l3 << std::endl; 

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

    /*
     *    CALCULATE IMAGE SQUARED (i.e. I * I)
     */

    Mat sqImage;
    createSquareImage(image, w_smooth, sqImage);

    int rows = sqImage.rows;
    int cols = sqImage.cols;
    int dim  = max(rows, cols);

    double normL = sqrt(sqr(l1) + sqr(l2) + sqr(l3));
    double L1 = l1 / normL;
    double L2 = l2 / normL;
    double L3 = l3 / normL;

    Point3d lightVector(L1,  L2,  L3);

    Point6dMatrix scMatrix(boost::extents[rows]);
    Point6dMatrix uTestMatrix(boost::extents[rows];

    Point3dMatrix normals_expected(boost::extents[rows]);

    generate_skin(sqImage, w_smooth, n_coeffs, generated_derivatives == 0, sc, si, sj, scMatrix, lightVector, uTestMatrix, normals_expected);

    /*
     *    CALCULATE LIGHT VECTORS
     */

    Point6dMatrix uMatrix(boost::extents[rows]);

    auto getter = [](Mat& m, int i, int j) {
        double get = m.at<unsigned short>(i, j) / 65025.0 /* 255x255 */;
        //std::cout << "get(" << i << ", " << j << ")=" << get << std::endl;
        return  get; 
    };

    std::cout << "starting second pass" << std::endl;

    int w = w_coeffs;
    int midi = rows / 2;
    int midj = cols / 2;

    pcl::PointCloud<pcl::PointXYZ> lightVectorCloud;
    lightVectorCloud.height   = 1;
    lightVectorCloud.is_dense = false;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {

             double x = scaleX(j, cols, dim);
             double y = scaleY(i, rows, dim);

             boost::optional<Point6d> u = calculate_normal_coefficients5(scMatrix, i, j, rows, cols, dim, w_coeffs, lastRow, lightVector);
             if (isSelectedPixel(i, j)) {
                 std::cout << "i =" << i << " j=" << j << " x=" << x << " y=" << y << std::endl;
                 std::cout << "s =" << scMatrix[i][j] << std::endl;
                 std::cout << "u =" << u << std::endl;
                 std::cout << "ut=" << uTestMatrix[i][j] << std::endl;
             }
             uMatrix[i][j] = u;
             if (!u) {
                 continue;
             }
             vector<Point3d> solutions;
             calculate_light_from_coefficients(u, scMatrix[i][j], solutions, x, y, i, j);
             for (int k = 0; k < solutions.size(); k++) {
                 boost::optional<pcl::PointXYZ> l = toLight(solutions[k]);

                 if (l) {
                     if (isSelectedPixel(i, j)) {
                          std::cout << "l=" << l << std::endl;
                     }
                     if (i >= li && i < (li + lw) && j >= lj && j < (lj + lw)) {
                     //  std::cout << "i=" << i << " j=" << j << " l=" << l << " solution=" << solutions[k] << std::endl;
                          lightVectorCloud.points.push_back(l.value());
                     }
                 }
             }
        }
    }	
    int totalLV = lightVectorCloud.points.size();
    if (totalLV > 0) {
        lightVectorCloud.width = totalLV;
        pcl::io::savePCDFileASCII ("pcds/L.pcd", lightVectorCloud);

        std::cout << "calculating light vector=" << lightVector << std::endl;

        Mat centers;
        Mat lightpoints(totalLV, 3, CV_32FC2), labels;
        for (int k = 0; k < totalLV; k++) {
            pcl::PointXYZ xyz = lightVectorCloud.points[k];
            lightpoints.at<float>(k, 0) = xyz.x;
            lightpoints.at<float>(k, 1) = xyz.y;
            lightpoints.at<float>(k, 2) = xyz.z;
        }

/*
        std::cout << "calculating kmeans of totalLV=" << totalLV << " points" << std::endl;
        kmeans(lightpoints, 5, labels, TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 4, 1.0 ), 4, KMEANS_PP_CENTERS, centers);

        std::cout << "print centers=" << centers.rows << " labels=" << labels.rows << std::endl;
        for (int k = 0; k < centers.rows; k++) {
            double c1 = centers.at<float>(k, 0);
            double c2 = centers.at<float>(k, 1);
            double c3 = centers.at<float>(k, 2);

            double normC = sqrt(sqr(c1) + sqr(c2) + sqr(c3));

            Point3d cNormed(c1 / normC,  c2 / normC,  c3 / normC);

            std::cout << "center=" << cNormed << std::endl;
        }
*/
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
        Point6d uv = uopt.value();//.normalize2();
        if (uv.x0 == 0 || uv.x1 == 0) {
             return 0.0;
        }
        //double u = uv.x0 * uv.x1 - sqr(uv.x2);
        double u = uv.x1 / uv.x0 - sqr(uv.x2 / uv.x0);
        //double u = uv.x0 * uv.x1;
        //double u = sqr(uv.x2);
        //double u = uv.x0 * uv.x1;
        if (abs(u) > 5) {
            return 0.0;
        }
        return  u; 
    };


    pcl::PointCloud<pcl::PointXYZRGB> uCloud;
    toCloud<Point6dMatrix>(uMatrix    , uCloud    , rows, cols, getterUMatrix);
    pcl::io::savePCDFileASCII ("pcds/U.pcd"    , uCloud);

    pcl::PointCloud<pcl::PointXYZ> uTestCloud;
    toCloud<Point6dMatrix>(uTestMatrix, uTestCloud, rows, cols, getterUMatrix);
    pcl::io::savePCDFileASCII ("pcds/UTest.pcd", uTestCloud);

}
