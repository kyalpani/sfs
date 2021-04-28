#include "point14d.h"

Point14d::Point14d() {
	this->x0 = 0;
	this->x1 = 0;
	this->x2 = 0;
	this->x3 = 0;
	this->x4 = 0;
	this->x5 = 0;
    this->x6 = 0;
    this->x7 = 0;
    this->x8 = 0;
    this->x9 = 0;
}

Point14d::Point14d(double x0, double x1, double x2, double x3, double x4, double x5, double x6, double x7, double x8, double x9, double x10, double x11, double x12, double x13, double x14) {
	this->x0 = x0;
	this->x1 = x1;
	this->x2 = x2;
	this->x3 = x3;
	this->x4 = x4;
	this->x5 = x5;
	this->x6 = x6;
	this->x7 = x7;
	this->x8 = x8;
	this->x9 = x9;
	this->x10 = x10;
	this->x11 = x11;
	this->x12 = x12;
	this->x13 = x13;
	this->x14 = x14;
}

double Point14d::eval(double dx, double dy) {
    double X = dx;
    double Y = dy;
    double X2 = sqr(X);
    double Y2 = sqr(Y);
	double X3 = X2 * X;
    double Y3 = Y2 * Y;
	double X4 = X2 * X2;
    double Y4 = Y2 * Y2;
    return this->x0 + this->x1 * X + this->x2 * Y + this->x3 * X2 + this->x4 * X * Y + this->x5 * Y2 + this->x6 * X3 + this->x7 * X2 * Y + this->x8 * X * Y2 + this->x9 * Y3 + this->x10 * X4 + this->x11 * X3 * Y + this->x12 * X2 * Y2 + this->x13 * X * Y3 + this->x14 * Y4;
}

double Point14d::evalDx(double dx, double dy) {
	double X = dx;
    double Y = dy;
    double X2 = sqr(X);
    double Y2 = sqr(Y);
	double X3 = X2 * X;
    double Y3 = Y2 * Y;
    return this->x1 + 2 * this->x3 * X + this->x4 * Y + 3 * this->x6 * X2 + 2 * this->x7 * X * Y + this->x8 * Y2 + 4 * this->x10 * X3 + 3 * this->x11 * X2 * Y + 2 * this->x12 * X * Y2 + this->x13 * Y3;
}

double Point14d::evalDy(double dx, double dy) {
    double X = dx;
    double Y = dy;
    double X2 = sqr(X);
    double Y2 = sqr(Y);
	double X3 = X2 * X;
    double Y3 = Y2 * Y;
	return this->x2 + this->x4 * X + 2 * this->x5 * Y + this->x7 * X2 + 2 * this->x8 * X * Y + 3 * this->x9 * Y2 + this->x11 * X3 + 2 * this->x12 * X2 * Y + 3 * this->x13 * X * Y2 + 4 * this->x14 * Y3;
}

double Point14d::evalDxx(double dx, double dy) {
	double X = dx;
    double Y = dy;
    double X2 = sqr(X);
    double Y2 = sqr(Y);
	return 2 * this->x3 + 6 * this->x6 * X + 2 * this->x7 * Y + 12 * this->x10 * X2 + 6 * this->x11 * X * Y + 2 * this->x12 * Y2;
}

double Point14d::evalDxy(double dx, double dy) {
	double X = dx;
    double Y = dy;
    double X2 = sqr(X);
    double Y2 = sqr(Y);
	return this->x4 + 2 * this->x7 * X + 2 * this->x8 * Y + 3 * this->x11 * X2 + 4 * this->x12 * X * Y + 3 * this->x13 * Y2;
}

double Point14d::evalDyy(double dx, double dy) {
	double X = dx;
    double Y = dy;
    double X2 = sqr(X);
    double Y2 = sqr(Y);
	return 2 * this->x5 + 2 * this->x8 * X + 6 * this->x9 * Y + 2 * this->x12 * X2 + 6 * this->x13 * X * Y + 12 * this->x14 * Y2;
}

std::ostream& operator<<(std::ostream& os, Point14d const& p) {
    os << "(" << p.x0 << "," << p.x1 << "," << p.x2 << "," << p.x3 << "," << p.x4 << "," << p.x5 << "," << p.x6 << "," << p.x7 << "," << p.x8 << "," << p.x9 << "," << p.x10 <<"," << p.x11 <<"," << p.x12 << "," << p.x13 <<"," << p.x14 <<")";
    return os;
}
