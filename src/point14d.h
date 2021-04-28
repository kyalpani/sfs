#ifndef __POINT14D_H__
#define __POINT14D_H__

#include "sfslib.h"

using namespace std;

class Point14d {

    public:

	    double x0;
	    double x1;
	    double x2;
	    double x3;
	    double x4;
	    double x5;
	    double x6;
	    double x7;
	    double x8;
	    double x9;
	    double x10;
	    double x11;
	    double x12;
	    double x13;
	    double x14;

	Point14d();
	Point14d(double x0, double x1, double x2, double x3, double x4, double x5, double x6, double x7, double x8, double x9, double x10, double x11, double x12, double x13, double x14);
	double eval(double dx, double dy);
    double evalDx(double dx, double dy);
    double evalDy(double dx, double dy);
    double evalDxx(double dx, double dy);
    double evalDxy(double dx, double dy);
    double evalDyy(double dx, double dy);
    friend std::ostream& operator<<(std::ostream& os, Point14d const& p);
};

std::ostream& operator<<(std::ostream& os, Point14d const& p);

#endif
