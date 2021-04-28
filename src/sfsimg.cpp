#include "boost/multi_array.hpp"

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/surface/mls.h>

#include <iostream>

#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <math.h>
#include <stdio.h>

using namespace cv;
using namespace std;


typedef boost::multi_array<double, 2> ZMatrix;
typedef ZMatrix::index rangeIndex;

static double scaleX(int j, int cols, int size) {
     return (j - cols / 2) * 1.0 / size;
}

static double scaleY(int i, int rows, int size) {
     return (i - rows / 2) * -1.0 / size;
}

template<class M> static void toCloud(M & m, pcl::PointCloud<pcl::PointXYZ> & cloud, std::function<double(M & m, int, int)> getter) {
    int rows = m.rows;
    int cols = m.rows;
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
             cloud.points[start + idx].z = getter(m, i, j);
        }
    }
}

template<class M> static void toCloud(M & m, pcl::PointCloud<pcl::PointXYZ> & cloud, std::function<double(M & m, int, int)> getter, int rows, int cols) {
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
             cloud.points[start + idx].z = getter(m, i, j);
        }
    }
}

static void toCloud(cv::Mat & m, pcl::PointCloud<pcl::PointXYZ> & cloud, std::function<double(cv::Mat& m, int, int)> getter) {
    int rows = m.rows;
    int cols = m.cols;
    int dim = max(rows, cols);
    cloud.width    = rows * cols;
    cloud.height   = 1;
    cloud.is_dense = false;
    cloud.points.resize (cloud.width * cloud.height);

    int idx = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++, idx++) {
             cloud.points[idx].x = scaleX(j, cols, dim);
             cloud.points[idx].y = scaleY(i, rows, dim);
             cloud.points[idx].z = getter(m, i, j);
        }
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
    "{help h||}{@image |sfsimg.jpeg|input image name}"
};

inline double sqr(double x) {
    return x * x;
}

int main( int argc, const char** argv )
{
    CommandLineParser parser(argc, argv, keys);
    string filename = parser.get<string>(0);
    std::cout << "filename:" << filename << std::endl;
    
    double l1 = 0.2;
    double l2 = 0.3;
    double l3 = 1;
    double normL = sqrt(l1 * l1 + l2 * l2 + l3 * l3);
    double a0 = -1;
    double a1 = -1;
    double a2 = 0;
    double a3 = 0;
    double a4 = 0;

    int rows = 800;
    int cols = 800;
    int dim =  max(rows, cols);
    cv::Mat image = cv::Mat::zeros(rows, cols, CV_16UC1);

    ZMatrix zMatrix(boost::extents[rows][cols]);
    double r = 0.5;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {

             double x = scaleX(j, cols, dim);
             double y = scaleY(i, rows, dim);
             //double y = 0;

             double z = 0;
             double p = std::numeric_limits<double>::infinity();
             double q = std::numeric_limits<double>::infinity();
             double n = sqr(x) + sqr(y);
             double norm = sqrt(n);
             if (norm <= r) {
                 double z = sqrt(sqr(r) - n);
                 p = -x / z;
                 q = -y / z;
             }

             zMatrix[i][j] = z;

             //p = 2 * a0 * x + a2 * y + a3;
             //q = 2 * a1 * y + a2 * x + a4;
             double numerator = - l1 * p - l2 * q + l3;
             double denuminator = sqrt(p * p + q * q + 1) * normL;
             double val = (numerator < 0) ? 0 : numerator / denuminator;
             int intVal = val * 255.0;
             if (i == 200 && j == 300) {
                 std::cout << "p=" << p << " q=" << q << std::endl;
                 std::cout << "i=" << i << " j=" << j << " intVal=" << intVal << " x=" << x << " y=" << y << " val=" << val  << std::endl;
             }
             image.at<unsigned short>(i, j) = intVal;
        }
    }
    imwrite(filename, image);

    pcl::PointCloud<pcl::PointXYZ> zCloud;
    auto getterZMatrix = [](ZMatrix& m, int i, int j) { return m[i][j]; };
    toCloud<ZMatrix>(zMatrix, zCloud, getterZMatrix, rows, cols);
    pcl::io::savePCDFileASCII ("Z.pcd", zCloud);

    return 0;
}
