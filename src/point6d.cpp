#include "point6d.h"


Point6d::Point6d() {
	this->x0 = 0;
	this->x1 = 0;
	this->x2 = 0;
	this->x3 = 0;
	this->x4 = 0;
	this->x5 = 0;
	this->rotation = 0;
}

Point6d::Point6d(double x0, double x1, double x2, double x3, double x4, double x5) {
	this->x0 = x0;
	this->x1 = x1;
	this->x2 = x2;
	this->x3 = x3;
	this->x4 = x4;
	this->x5 = x5;
	this->rotation = 0;
}

Point6d Point6d::scale(double factor) {
	Point6d scaled(this->x0 * factor, this->x1 * factor, this->x2 * factor, this->x3 * factor, this->x4 * factor, this->x5 * factor);
	scaled.rotation = this->rotation * sgn(factor);
	scale_ = factor;
	return scaled;
}

Point6d Point6d::sum(Point6d o) {
	return Point6d(this->x0 + o.x0, this->x1 + o.x1, this->x2 + o.x2, this->x3 + o.x3, this->x4 + o.x4, this->x5 + o.x5);
}

Point6d Point6d::diff(Point6d o) {
	return Point6d(this->x0 - o.x0, this->x1 - o.x1, this->x2 - o.x2, this->x3 - o.x3, this->x4 - o.x4, this->x5 - o.x5);
}

double Point6d::get(int idx) {
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

double Point6d::len() {
	return sqrt(sqr(this->x0) + sqr(this->x1) + sqr(this->x2) + sqr(this->x3) + sqr(this->x4) + sqr(this->x5));
}

Point6d Point6d::put(int idx, double val) {
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


double Point6d::dot(Point6d o) {
    return (this->x0 * o.x0 + this->x1 * o.x1 + this->x2 * o.x2 + this->x3 * o.x3 + this->x4 * o.x4 + this->x5 * o.x5 );
}

Point6d Point6d::getTranslated(double x, double y) {
	double c0 = this->x0 - this->x1 * x - this->x2 * y + this->x3 * sqr(x) + this->x4 * x * y + this->x5 * sqr(y);
    double c1 = this->x1 -  2 * this->x3 * x - this->x4 * y;
	double c2 = this->x2 -  2 * this->x5 * y - this->x4 * x;
	return Point6d(c0, c1, c2, this->x3, this->x4, this->x5);
}

Point6d Point6d::rotateBy(double theta) {
	double sint = sin(theta);
	double cost = cos(theta);
	double sin2t = sin(2 * theta);
	double cos2t = cos(2 * theta);
	double sint2 = sqr(sint);
	double csint = cost * sint;
	double cost2 = sqr(cost);

	Point6d rotated(this->x0,
			       this->x1 * cost + this->x2 * sint,
			       this->x2 * cost - this->x1 * sint,
                   this->x3 * cost2 + this->x4 * csint + this->x5 * sint2,
				   this->x4 * cos2t + (this->x5 - this->x3) * sin2t,
				   this->x3 * sint2 - this->x4 * csint + this->x5 * cost2);
	rotated.rotation = this->rotation + theta;
	return rotated;
}

Point6d Point6d::rotateToZero() {
	double sint = this->x4;
	double cost = normalize0(this->x3 - this->x5);
	double t = cost == 0 ? sgn(sint) * PI / 2 : atan(sint / cost) / 2;
    return this->rotateBy(t);
}

double Point6d::getRotation() {
	return this->rotation;
}

double Point6d::getRotationDegrees() {
	return this->rotation * 180 / 3.1415439;
}

Point6d Point6d::getU(double x, double y, Point3d & lv) {
	double c0 = this->x0;
    double c1 = this->x1;
	double c2 = this->x2;
    double lambda = this->x3 + this->x5;
    return Point6d(normalize0(c0 + 2 * c1 * x + lambda * sqr(x) - sqr(lv.x)),
                   normalize0(c0 + 2 * c2 * y + lambda * sqr(y) - sqr(lv.y)),
                   normalize0(c1 * y + c2 * x + lambda * x * y - lv.x * lv.y),
                   normalize0(c1 + lambda * x),
                   normalize0(c2 + lambda * y),
                   normalize0(lambda)
                 );
}

Point6d Point6d::getU0(double x, double y) {

   return Point6d(normalize0(this->x0 + this->x1 * x + this->x2 * y + this->x3 * sqr(x) + this->x4 * x * y + this->x5 * sqr(y)),
                  normalize0(this->x1 + 2 * this->x3 * x + this->x4 * y),
                  normalize0(this->x2 + 2 * this->x5 * y + this->x4 * x),
                  normalize0(this->x3),
                  normalize0(this->x4),
                  normalize0(this->x5));
}

Point6d Point6d::getU1(double x, double y) {
	   return Point6d(normalize0(this->x0 -     this->x1 * x - this->x2 * y - this->x3 * sqr(x) - this->x4 * x * y - this->x5 * sqr(y)),
	                  normalize0(this->x1 - 2 * this->x3 * x - this->x4 * y),
	                  normalize0(this->x2 - 2 * this->x5 * y - this->x4 * x),
	                  normalize0(this->x3),
	                  normalize0(this->x4),
	                  normalize0(this->x5));
}

Point6d Point6d::getU2(double x, double y) {
   return Point6d(normalize0(2 * this->x2 * x + this->x4 * sqr(x)),
                  normalize0(2 * this->x1 * y + this->x4 * sqr(y)),
                  normalize0(this->x0 + this->x1 * x + this->x2 * y + this->x4 * x * y),
                  normalize0(this->x2 + this->x4 * x),
                  normalize0(this->x1 + this->x3 * y),
                  normalize0(this->x4));
}

Point6d Point6d::getU3(double x, double y) {
    return Point6d(normalize0(2 * this->x0 * x + this->x1 * sqr(x)),
              normalize0(                this->x1 * sqr(y)),
              normalize0(    this->x0 * y + this->x1 * x * y),
              normalize0(        this->x0     + this->x1 * x),
              normalize0(                    this->x1 * y),
              normalize0(                       this->x1));
}

Point6d Point6d::getU4(double x, double y) {
   return Point6d(normalize0(                this->x2 * sqr(x)),
                  normalize0(2 * this->x0 * y + this->x2 * sqr(y)),
                  normalize0(    this->x0 * x + this->x2 * x * y),
                  normalize0(                    this->x2 * x),
                  normalize0(        this->x0     + this->x2 * y),
                  normalize0(                        this->x2));
}

Point6d Point6d::getU5(double x, double y) {
   return Point6d(normalize0(sqr(x)),
                  normalize0(sqr(y)),
                  normalize0(x * y),
                  normalize0(x),
                  normalize0(y),
                  normalize0(1));
}

gsl_vector * Point6d::to_gsl_vector() {
    gsl_vector * vec = gsl_vector_alloc(6);
    gsl_vector_set(vec, 0, this->x0);
    gsl_vector_set(vec, 1, this->x1);
    gsl_vector_set(vec, 2, this->x2);
    gsl_vector_set(vec, 3, this->x3);
    gsl_vector_set(vec, 4, this->x4);
    gsl_vector_set(vec, 5, this->x5);
    return vec;
}


void Point6d::write_data_file(std::ofstream & str, int i, int j) {
    str << i << " " << j << " " << this->x0 << " " << this->x1 << " " << this->x2 << " " << this->x3 << " " << this->x4 << " " << this->x5 << std::endl;
}

std::ostream& operator<<(std::ostream& os, Point6d const& p) {
    os << "(" << p.x0 << "," << p.x1 << "," << p.x2 << "," << p.x3 << "," << p.x4 << "," << p.x5 << ")";
    return os;
}
