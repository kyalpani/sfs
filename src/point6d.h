#ifndef __POINT6D_H__
#define __POINT6D_H__

#include "sfslib.h"

using namespace std;
using namespace cv;

class Point6d {

    public:

	    double x0;
	    double x1;
	    double x2;
	    double x3;
	    double x4;
	    double x5;
	    double scale_ = 1;

	Point6d();
	Point6d(double x0, double x1, double x2, double x3, double x4, double x5);
	Point6d scale(double factor);
	Point6d sum(Point6d o);
	Point6d diff(Point6d o);
	double get(int idx);
	Point6d put(int idx, double val);
    double dot(Point6d o);
    double len();
    Point6d getTranslated(double x, double y);
    Point6d rotateBy(double theta);
    Point6d rotateToZero();
    double getRotation();
    double getRotationDegrees();
    double hFactor();
    Point6d getU(double x, double y, Point3d & lv);
    Point6d getU0(double x, double y);
    Point6d getU1(double x, double y);
    Point6d getU2(double x, double y);
    Point6d getU3(double x, double y);
    Point6d getU4(double x, double y);
    Point6d getU5(double x, double y);
    gsl_vector * to_gsl_vector();
    void write_data_file(std::ofstream & s, int i, int j);

	friend std::ostream& operator<<(std::ostream& os, Point6d const& p);

    private:
	    double rotation = 0;
};

std::ostream& operator<<(std::ostream& os, Point6d const& p);
#endif
