#include "boost/multi_array.hpp"

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
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

using namespace cv;
using namespace std;

const double EPSILON_ZERO = 0.1;
double mm_average = 0;
int mm_count = 0;

int di=0;
int dj=0;
int dw=0;

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
*/

/*
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
     return true;
}

static inline double sqr(double x) {
    return x * x;
}

static double roundd(double x) {
    return round(x * 100) / 100;
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

static bool is0(double num, double epsilon) {
    return abs(num) < epsilon;
}

static bool is0(double num) {
    return num == 0; //is0(num, EPSILON_ZERO);
}

static bool leq0(double num) {
    return abs(num) < EPSILON_ZERO || num < 0;
}

inline static double normalize0(double x) {
    return abs(x) < 0.0001 ? 0 : x;
}

static void createSmoothImage(Mat & m, int w_smooth, Mat & smooth) {
    Mat gray;
    cvtColor(m, gray, COLOR_BGR2GRAY);
    blur( gray, smooth, Size(5, 5));
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

static void calculate_skin_test(Mat& sqImage, Point6dMatrix & scMatrix) {
    int rows = sqImage.rows;
    int cols = sqImage.cols;
    int dim  = max(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double x = scaleX(j, cols, dim);
            double y = scaleY(i, rows, dim);
   
            double l1pl2q1 =  -0.4 * x - 0.4 * y + 1;
            double p2q21   = 4 * x * x + 4 * y * y + 1;
            double J = .93 * l1pl2q1 *  l1pl2q1 / p2q21;
            double Jx1 = - 0.74 * l1pl2q1 / p2q21 - 7.4 * x * l1pl2q1 * l1pl2q1 / (p2q21 * p2q21);
            double Jy1 = - 0.74 * l1pl2q1 / p2q21 - 7.4 * y * l1pl2q1 * l1pl2q1 / (p2q21 * p2q21);

            double Jx2 = 0.3 / p2q21 + 11.85 * x * l1pl2q1 / (p2q21 * p2q21) - 7.4 * l1pl2q1 * l1pl2q1 / (p2q21 * p2q21) + 118.5 * x * x * l1pl2q1 * l1pl2q1 / (p2q21 * p2q21 * p2q21);
            double Jy2 = 0.3 / p2q21 + 11.85 * y * l1pl2q1 / (p2q21 * p2q21) - 7.4 * l1pl2q1 * l1pl2q1 / (p2q21 * p2q21) + 118.5 * y * y * l1pl2q1 * l1pl2q1 / (p2q21 * p2q21 * p2q21);

            double Jxy = 0.3 / p2q21 + 5.93 * y * l1pl2q1 / (p2q21 * p2q21) + 5.93 * x * l1pl2q1 / (p2q21 * p2q21)       + 118.5 * x * y * l1pl2q1 * l1pl2q1 / (p2q21 * p2q21 * p2q21);


            Point6d s(J , Jx1, Jy1, Jx2 / 2, Jxy / 2, Jy2 / 2);
            scMatrix[i][j] = s;
        }
    }
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


static void generate_skin(Mat & sqImage, int w_smooth, int n_coeffs, bool use_real_data, Point6dMatrix & scMatrix) {
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
        calculate_skin_test(sqImage, scMatrix);
    }    
}

static boost::optional<Point6d> getFromScMatrix(Point6dMatrix& scMatrix, int i, int j, int rows, int cols, int dim, int w) {

    bool iInBounds = (i >= w && i < (rows - w));
    bool jInBounds = (j >= w && j < (cols - w));

    if (isSelectedPixel(i, j)) {
        std::cout << "getFromScMatrix i=" << i << " j=" << j << " iInBounds=" << iInBounds << " jInBounds=" << jInBounds << std::endl;
    }

    if (iInBounds && jInBounds) {
        return scMatrix[i - w][j - w];
    }

    boost::optional<Point6d> popt = {};
    bool usei = false;

    int tmpi;
    int tmpj;

    if (!iInBounds && !jInBounds) {

       return {};
/*
        if (i < 0 && j < 0) {
            if (abs(i) <= abs(j)) {
                tmpi = 0;
                tmpj = j;
                usei = true;
            } else {
                tmpi = i;
                tmpj = 0;
            }
        }

        if (i >= rows && j < 0) {
            if (abs(i - rows) <= abs(j)) {
               tmpi = rows - 1;
               tmpj = j;
               usei = true;
            } else {
               tmpi = i;
               tmpj = 0;
            }
        }

        if (i < 0 && j >= cols) {
            if (abs(i) <= abs(j - cols)) {
                tmpi = 0;
                tmpj = j;
                usei = true;
            } else {
                tmpi = i;
                tmpj = cols - 1;
            }
        }

        if (i >= rows && j >= cols) {
            if (abs(i - rows) <= abs(j - cols)) {
                tmpi = rows - 1;
                tmpj = j;
                usei = true;
            } else {
                tmpi = i;
                tmpj = cols - 1;
            }
        }

        popt = getFromScMatrix(scMatrix, tmpi, tmpj, rows, cols, dim, w);
*/
    } else {
        if (iInBounds) {
           tmpi = i - w;
           tmpj = j < w ? 0 : cols - 2 * w - 1;
        } else {
           tmpi = i < w ? 0 : rows - 2 * w - 1;
           tmpj = j - w;
           usei = true;
        }

    }

    if (isSelectedPixel(i, j)) {
        std::cout << "i=" << i << " j=" << j << " tmpi=" << tmpi << " tmpj=" << tmpj << " popt=" << popt << std::endl;
    }
    popt = scMatrix[tmpi][tmpj];

//             [i][j] = Point6d(J, vx.y + 2 * vx.z * x, vy.y + 2 * vy.z * y, vx.z / 2 , 0, vy.z / 2);


    if (!popt) {
        return {};
    }

    Point6d p    = popt.value();

    double x0    = scaleX(tmpj, cols, dim);
    double y0    = scaleY(tmpi, rows, dim);

    double x     = scaleX(j, cols, dim);
    double y     = scaleY(i, rows, dim);

    double z     = usei ? (p.x0 + p.x2 * (y - y0) + p.x5 * sqr(y - y0)) : (p.x0 + p.x1 * (x - x0) + p.x3 * sqr(x - x0));

    double dzdx1 = p.x1 + 2 * p.x3 * (x - x0);
    double dzdy1 = p.x2 + 2 * p.x5 * (y - y0);

    Point6d virt(z, dzdx1, dzdy1, p.x3, 0, p.x5);
    if (isSelectedPixel(i, j)) {
        std::cout << "p=" << p << " virt=" << virt << " x=" << x << " y=" << y << " x0=" << x0 << " y0=" << y0 <<  std::endl;
    }
    return virt;
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
    "{@image                      |sfs.png     | input image name}"
    "{out                         |sfs_out.png | input image name}"
    "{n_coeffs              count | 10         | number of bspline coefficients}"
    "{w_coeffs              count | 6          | window size for calculation normal coefficients}"
    "{w_smooth              count | 7          | filter window size for smoothing}"
    "{generated_derivatives count | 0          | 0 if derivatives are generated 1 if they are computed from input data }"
    "{di                    count | 0          | the i-index of the debug pixel }"
    "{dj                    count | 0          | the j-index of the debug pixel }"
    "{dw                    count | 0          | the window size of debug pixel window }"
};

static double uval(boost::optional<Point6d> u) {
    return u ? (u.value().x0 * u.value().x1 - sqr(u.value().x2)) : 0;
}


int main( int argc, const char** argv ) {

    /*
     *    PARSE COMMAND LINE
     */

    CommandLineParser parser(argc, argv, keys);
    string filename = parser.get<string>(0);
    Mat preImage = imread(filename, IMREAD_COLOR);
    if(preImage.empty())
    {
        printf("Cannot read image file: %s\n", filename.c_str());
        help();
        return -1;
    }
    std::cout << "image size:" << preImage.size() << std::endl; 

    String out = parser.get<String>("out");
    std::cout << "out:" << out << std::endl;     

    int n_coeffs = parser.get<int>("n_coeffs");
    std::cout << "n_coeffs:" << n_coeffs << std::endl;     

    int w_coeffs = parser.get<int>("w_coeffs");
    std::cout << "w_coeffs:" << w_coeffs << std::endl; 

    int w_smooth = parser.get<int>("w_smooth");
    std::cout << "w_smooth:" << w_smooth << std::endl; 

    int generated_derivatives = parser.get<int>("generated_derivatives");
    std::cout << "generated_derivatives:" << generated_derivatives << std::endl; 

    di = parser.get<int>("di");
    std::cout << "di:" << di << std::endl; 

    dj = parser.get<int>("dj");
    std::cout << "dj:" << dj << std::endl; 

    dw = parser.get<int>("dw");
    std::cout << "dw:" << dw << std::endl; 

    /*
     *    CALCULATE IMAGE SQUARED (i.e. I * I)
     */

    createSmoothImage(preImage, w_smooth, image);
    imwrite("preImage.png", preImage);

    int rows = image.rows;
    int cols = image.cols;

    Point6dMatrix scMatrix(boost::extents[rows][cols]);
    generate_skin(preImage, w_smooth, n_coeffs, generated_derivatives == 0, scMatrix);

    Mat imag_orig = cv::Mat::zeros(rows, cols, CV_8U);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            imag_orig.at<uchar>(i, j) = 0;
            boost::optional<Point6d> sc = scMatrix[i][j];
            if (!sc) {
                continue;
            }
            double I = sc.value().x0;

            imag_orig.at<uchar>(i, j) = I * 255;
            std::cout << "i=" << i << " j=" << j << " I=" << I << " M=" << imag_orig.at<uchar>(i, j) << std::endl;
            
        }
    }
    imwrite("test.png", imag_orig);


    int rows_ext = rows + 2 * w_coeffs;
    int cols_ext = cols + 2 * w_coeffs;
    int dim  = max(rows_ext, cols_ext);
    Mat imag_exp = cv::Mat::zeros(rows_ext, cols_ext, CV_8UC1);
    for (int i = 0; i < rows_ext; i++) {
        for (int j = 0; j < cols_ext; j++) {
            imag_exp.at<uchar>(i, j) = 0;
            boost::optional<Point6d> sc = getFromScMatrix(scMatrix, i, j, rows_ext, cols_ext, dim, w_coeffs);
            if (!sc) {
                continue;
            }
            double I = sc.value().x0;

            imag_exp.at<uchar>(i, j) = abs(I * 255);
            std::cout << "i=" << i << " j=" << j << " I=" << I << " M=" << imag_exp.at<unsigned short>(i, j) << std::endl;
            
        }
    }
    std::cout << "write image out=" << out << std::endl;
    imwrite(out, imag_exp);
}
