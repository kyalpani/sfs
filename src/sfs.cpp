#include "sfs.h"
#include "sfslib.h"

using namespace std;
using namespace cv;
using namespace pcl;

#include "point14d.h"
#include <ginac/ginac.h>

SFS::SFS(int argc, const char **argv) {
	/*
	 *    PARSE COMMAND LINE
	 */

	CommandLineParser parser(argc, argv, keys);

	imgFile = parser.get<string>(0);
	skin = getParmString("skin", parser);
	n_coeffs = getParmInt("n_coeffs", parser);
	w_coeffs = getParmInt("w_coeffs", parser);
	generated_derivatives = getParmInt("generated_derivatives", parser);
	si = getParmInt("si", parser);
	sj = getParmInt("sj", parser);
	li = getParmInt("li", parser);
	lj = getParmInt("lj", parser);
	lw = getParmInt("lw", parser);
	llw = getParmInt("llw", parser);
	di = getParmInt("di", parser);
	dj = getParmInt("dj", parser);
	dw = getParmInt("dw", parser);
	blocksize = getParmInt("blocksize", parser);
	test = getParmInt("test", parser);
	l1 = getParmDouble("l1", parser);
	l2 = getParmDouble("l2", parser);
	l3 = getParmDouble("l3", parser);
	testcoeff1 = getParmDouble("testcoeff1", parser);
	testcoeff2 = getParmDouble("testcoeff2", parser);
	testcoeff3 = getParmDouble("testcoeff3", parser);
	testcoeff4 = getParmDouble("testcoeff4", parser);
	testcoeff5 = getParmDouble("testcoeff5", parser);
	testcoeff6 = getParmDouble("testcoeff6", parser);
	epsilon_zero = getParmDouble("epsilon", parser);
	epsilon_m0 = getParmDouble("epsilon_m0", parser);
	cuti = getParmInt("cuti", parser);
	cutj = getParmInt("cutj", parser);
	z0height = getParmInt("z0height", parser);
	pGetter0 = getParmInt("pGetter0", parser);
	qGetter0 = getParmInt("qGetter0", parser);
	pGetter = getParmInt("pGetter", parser);
	qGetter = getParmInt("qGetter", parser);
	computeLight = getParmInt("computeLight", parser);
	getter_upper_limit = getParmInt("getter_upper_limit", parser);
	lastCol = getParmInt("lastCol", parser);
	detLimitLow = getParmInt("detLimitLow", parser);
	detLimitHigh = getParmInt("detLimitHigh", parser);

	lv = unit(Point3d(l1, l2, l3));
	std::cout << "lv: " << lv << std::endl;

	makeI2();

	cx = centerX();
	cy = centerY();
}

double SFS::getter(int type, Point6dMatrix &m, int i, int j) {

	boost::optional<Point6d> sopt = m[i][j];
	if (!sopt) {
		return 0.0;
	}

	double x = scaleX(j);
	double y = scaleY(i);
	Point6d s = sopt.value();
	double val;
	switch (type) {
	case 0:
		return thresh(s.x0, getter_upper_limit);
	case 1:
		return thresh(s.x1, getter_upper_limit);
	case 2:
		return thresh(s.x2, getter_upper_limit);
	case 3:
		return thresh(s.x3, getter_upper_limit);
	case 4:
		return thresh(s.x4, getter_upper_limit);
	case 5:
		return thresh(s.x5, getter_upper_limit);
	case 6:
		return s.x0 * s.x1 - sqr(s.x2);
	case 7:
		return thresh(s.x5, getter_upper_limit);
	case 8:
		return thresh(s.x3 * sqr(x) + s.x5 * sqr(y) + s.x4 * x * y + s.x1 * x + s.x1 * y, getter_upper_limit);
	case 9:
		return thresh(2 * s.x3 * x + s.x4 * y + s.x1, getter_upper_limit);
	case 10:
		return thresh(2 * s.x5 * y + s.x4 * x + s.x2, getter_upper_limit);
	case 11:
		return thresh(s.x3 * sqr(x) + s.x5 * sqr(y) + s.x4 * x * y + s.x1 * x + s.x2 * y + s.x0 + 1, getter_upper_limit);
	case 12:
		return thresh(s.x1, getter_upper_limit);
	case 13:
		return thresh(s.x2, getter_upper_limit);
	case 14:
		return thresh(s.x3, getter_upper_limit);
	case 15:
		return thresh(s.x4, getter_upper_limit);
	case 16:
		return thresh(sqr(s.x0), getter_upper_limit);
	}
	return s.x0;
}
;

double SFS::getter(int type, HeightMatrix &m, int i, int j) {

	boost::optional<double> sopt = m[i][j];
	if (!sopt || abs(sopt.value()) > 2) {
		return 0.0;
	}
	return sopt.value();
}
;

bool SFS::isSelectedPixel(int i, int j) {
	return (i == di && j == dj)
			|| (dw > 0
					&& ((i == (di + dw) && j == (dj + dw)) || (i == (di + dw) && j == (dj - dw)) || (i == (di - dw) && j == (dj + dw)) || (i == (di - dw) && j == (dj - dw))
							|| (i == di && j == (dj - dw)) || (i == di && j == (dj + dw)) || (i == (di + dw) && j == dj) || (i == (di - dw) && j == dj)));
}

bool SFS::isSelectedPixel() {
	return ((selected_i == di) || (selected_i == -1)) && ((selected_j == dj) || (selected_j == -1));
}

void SFS::markSelectedPixel(int i, int j) {
	sel_i = selected_i = i;
	sel_j = selected_j = j;
}

void SFS::markStep(String step) {
	if (isSelectedPixel()) {
		for (int i = 0; i < step.length(); i++) {
			std::cout << "=";
		}
		std::cout << std::endl;
		std::cout << step << std::endl;
		for (int i = 0; i < step.length(); i++) {
			std::cout << "=";
		}
		std::cout << std::endl;
	}
}

void SFS::markStepAlways(String step) {
	for (int i = 0; i < step.length(); i++) {
		std::cout << "=";
	}
	std::cout << std::endl;
	std::cout << step << std::endl;
	for (int i = 0; i < step.length(); i++) {
		std::cout << "=";
	}
	std::cout << std::endl;
}

void SFS::makeI2() {
	Mat I = imread(imgFile, IMREAD_COLOR);
	if (I.empty()) {
		printf("Cannot read image file: %s\n", imgFile.c_str());
		help();
		exit(EXIT_FAILURE);
	}

	/*
	 *    CALCULATE IMAGE SQUARED (i.e. I * I)
	 */
	rows = I.rows;
	cols = I.cols;
	dim = max(rows, cols);

	std::map<int, int> factorsRows;
	getPrimeFactors(rows, factorsRows);
	std::cout << "getPrimeFactors for rows=" << rows << std::endl;
	printPrimeFactors(factorsRows);

	std::map<int, int> factorsCols;
	getPrimeFactors(cols, factorsCols);
	std::cout << "getPrimeFactors for cols=" << cols << std::endl;
	printPrimeFactors(factorsCols);

	// the gcd value subdivides the image into a grid
	// of equal number of rows and columns, i.e. a square
	// grid. Each grid cell
	// consist of blocks of size rows/gcd X cols/gcd pixels
	// if the aspect ratio of the image == 1 then initially
	// each block will have (according to this determination)
	// only 1 pixel (e.g. with a 4:3 aspect ratio it will have 12 pixels).
	std::map<int, int> gcd;
	gcdPrimeFactors(factorsRows, factorsCols, gcd);
	std::cout << "gcd rows to cols=" << std::endl;
	printPrimeFactors(gcd);

	normativeSize = mergePrimeFactors(gcd);
	std::cout << "merge gcd=" << normativeSize << std::endl;
	aspecth = rows / normativeSize;
	aspectw = cols / normativeSize;

	int split = splitPrimeFactors(gcd, aspecth, aspectw, blocksize);
	aspecth *= split;
	aspectw *= split;

	std::cout << "split=" << split << " aspecth=" << aspecth << " aspectw=" << aspectw << std::endl;

	Mat gray;
	cvtColor(I, gray, COLOR_BGR2GRAY);

	I2.create(I.size(), CV_16UC1);
	// CALCULATE IMAGE SQUARED (i.e. I * I)
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int val = gray.at<uchar>(i, j);
			I2.at<unsigned short>(i, j) = val * val;
		}
	}
}

double SFS::scaleX(int j) {
	return (j - cols / 2) * 1.0 / dim;
}

double SFS::scaleY(int i) {
	return (i - rows / 2) * -1.0 / dim;
}

double SFS::getX() {
	return scaleX(selected_j);
}

double SFS::getY() {
	return scaleY(selected_i);
}

double SFS::getDeltaX() {
	return 1.0 / cols;
}

double SFS::getDeltaY() {
	return 1.0 / rows;
}

double SFS::centerX() {
	int base = cols / 2;
	if (cols % 2 == 0) {
		return (scaleX(base) + scaleX(base + 1)) / 2;
	} else {
		return scaleX(base + 1);
	}
}

double SFS::centerY() {
	int base = rows / 2;
	if (rows % 2 == 0) {
		return (scaleY(base) + scaleY(base + 1)) / 2;
	} else {
		return scaleY(base + 1);
	}
}

double SFS::getPixelValue(int i, int j) {
	return I2.at<unsigned short>(i, j) / 65025.0;
}

double SFS::getSelectedValue() {
	return I2.at<unsigned short>(selected_i, selected_j) / 65025.0;
}

bool SFS::cut() {
	if (selected_i == -1 || selected_j == -1) {
		return false;
	}
	if (cuti == -1 && cutj == -1) {
		return false;
	}
	if (cuti != -1 && cutj != -1) {
		return false;
	}
	if (cuti != -1) {
		return selected_i != cuti;
	}
	if (cutj != -1) {
		return selected_j != cutj;
	}

	return false;
}

bool SFS::cutI() {
	if (selected_i == -1 || cuti == -1) {
		return false;
	}
	return selected_i == cuti;
}

bool SFS::cutJ() {
	if (selected_j == -1 || cutj == -1) {
		return false;
	}
	return selected_j == cutj;
}

bool SFS::isBorderPixel() {
	return selected_i < w_coeffs || selected_j < w_coeffs || selected_i >= (rows - w_coeffs) || selected_j >= (cols - w_coeffs);
}

void SFS::calculate_skin_test(std::function<GiNaC::ex(GiNaC::symbol x, GiNaC::symbol y)> zFunction, std::function<bool(double x, double y)> zDomain) {

	double l1 = lv.x;
	double l2 = lv.y;
	double l3 = lv.z;

	GiNaC::symbol x("x");
	GiNaC::symbol y("y");

	GiNaC::ex z = zFunction(x, y);

	GiNaC::ex p = z.diff(x);
	GiNaC::ex q = z.diff(y);

	GiNaC::ex Px = p.diff(x);
	GiNaC::ex Py = p.diff(y);
	GiNaC::ex Qx = q.diff(x);
	GiNaC::ex Qy = q.diff(y);

	GiNaC::ex I = (-l1 * p - l2 * q + l3) / sqrt(GiNaC::pow(p, 2) + GiNaC::pow(q, 2) + 1);
	GiNaC::ex JJ = I * I;
	GiNaC::ex Jx = JJ.diff(x);
	GiNaC::ex Jy = JJ.diff(y);
	GiNaC::ex Jxx = Jx.diff(x);
	GiNaC::ex Jxy = Jx.diff(y);
	GiNaC::ex Jyy = Jy.diff(y);

	tmpMatrix.resize(boost::extents[rows][cols]);
	std::ofstream s_file(skin + "/s.txt");
	std::ofstream u_file(skin + "/u.txt");
	std::ofstream h_file(skin + "/h.txt");
	std::ofstream m_file(skin + "/m.txt");
	std::ofstream pq_file(skin + "/pq.txt");
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {

			markSelectedPixel(i, j);
			sMatrix[i][j] = { };
			hTestMatrix[i][j] = { };
			pqMatrix[i][j] = { };
			uTestMatrix[i][j] = { };
			tmpMatrix[i][j] = { };

			double X = getX();
			double Y = getY();

			if (!zDomain(X, Y)) {
				continue;
			}

			try {
				double h3 = evalXY(Px, x, y, X, Y) / 2;
				double h5 = evalXY(Qy, x, y, X, Y) / 2;
				double h4 = evalXY(Py, x, y, X, Y);

				double p_ = evalXY(p, x, y, X, Y);
				double q_ = evalXY(q, x, y, X, Y);

				double h1 = p_ - 2 * h3 * X - h4 * Y;
				double h2 = q_ - 2 * h5 * Y - h4 * X;
				double h0 = evalXY(z, x, y, X, Y);

				Point6d h(h0, h1, h2, h3, h4, h5);
				hTestMatrix[i][j] = h;

				double d0 = 2 * l1 * h3 + l2 * h4;
				double d1 = 2 * l2 * h5 + l1 * h4;
				double d2 = l1 * h1 + l2 * h2 - l3;

				double M_N = d2 + X * d0 + Y * d1;
				Point6d m(sqr(M_N), 2 * (M_N) * d0, 2 * (M_N) * d1, sqr(d0), 2 * d0 * d1, sqr(d1));
				mTestMatrix[i][j] = m;

				double u0 = sqr(h1) + sqr(h2) + 1;
				double u1 = 4 * h3 * h1 + 2 * h4 * h2;
				double u2 = 2 * h4 * h1 + 4 * h5 * h2;
				double u3 = 4 * sqr(h3) + sqr(h4);
				double u4 = 4 * h4 * (h3 + h5);
				double u5 = 4 * sqr(h5) + sqr(h4);

				double px_ = evalXY(Px, x, y, X, Y);
				double py_ = evalXY(Py, x, y, X, Y);
				double qx_ = evalXY(Qx, x, y, X, Y);
				double qy_ = evalXY(Qy, x, y, X, Y);

				double I_ = evalXY(I, x, y, X, Y);
				double J_ = evalXY(JJ, x, y, X, Y);
				double Jx_ = evalXY(Jx, x, y, X, Y);
				double Jy_ = evalXY(Jy, x, y, X, Y);
				double Jxx_ = evalXY(Jxx, x, y, X, Y);
				double Jxy_ = evalXY(Jxy, x, y, X, Y);
				double Jyy_ = evalXY(Jyy, x, y, X, Y);

				Point6d s(J_, Jx_, Jy_, Jxx_ / 2, Jxy_, Jyy_ / 2);

				sMatrix[i][j] = s;

				normals_expected[i][j] = Point3d(-p_, -q_, 1);
				Point6d pq(p_, q_, 0, 0, 0, 0);
				pqMatrix[i][j] = pq;
				Point6d u = Point6d(u0, u1, u2, u3, u4, u5);
				uTestMatrix[i][j] = u;

				double hessian = s.x3 * s.x5 - sqr(s.x4);
				//tmpMatrix[i][j] = Point6d(0, sqrt(sqr(s.x1) + sqr(s.x2)), hessian <= 0 ? 0 : log(hessian), 0.0, 0.0, 0.0);
				tmpMatrix[i][j] = Point6d(0, sqrt(sqr(s.x1) + sqr(s.x2)), hessian, 0.0, 0.0, 0.0);

				s.write_data_file(s_file, i, j);
				u.write_data_file(u_file, i, j);
				h.write_data_file(h_file, i, j);
				m.write_data_file(m_file, i, j);
				pq.write_data_file(pq_file, i, j);

				if (isSelectedPixel(i, j)) {
					std::cout << "Jx fn=" << Jx << std::endl;
					std::cout << "Jy fn=" << Jy << std::endl;

					std::cout << "i=" << i << " j=" << j << std::endl;
					std::cout << "X=" << X << " Y=" << Y << " before" << std::endl;
					std::cout << "z=" << hTestMatrix[i][j].value().x5 << std::endl;

					std::cout << "p =" << p_ << std::endl;
					std::cout << "q =" << q_ << std::endl;
					std::cout << "2 * h3 * X - h4 * Y =" << (2 * h3 * X + h4 * Y) << std::endl;
					std::cout << "2 * h5 * Y - h4 * X =" << (2 * h5 * Y + h4 * X) << std::endl;
					std::cout << "detH=" << (4 * h3 * h5 - sqr(h4)) << std::endl;
					std::cout << "h1 =" << h1 << std::endl;
					std::cout << "h2 =" << h2 << std::endl;

					std::cout << "px=" << px_ << std::endl;
					std::cout << "py=" << py_ << std::endl;
					std::cout << "qx=" << qx_ << std::endl;
					std::cout << "qy=" << qy_ << std::endl;

					std::cout << "J=" << J_ << std::endl;
					std::cout << "Jx=" << Jx_ << std::endl;
					std::cout << "Jy=" << Jy_ << std::endl;
					std::cout << "Jxx=" << Jxx_ << std::endl;
					std::cout << "Jxy=" << Jxy_ << std::endl;
					std::cout << "Jyy=" << Jyy_ << std::endl;

					std::cout << "s=" << s << std::endl;
					std::cout << "a=" << h << std::endl;
					std::cout << "u=" << uTestMatrix[i][j].value() << std::endl;
					std::cout << "n=" << normals_expected[i][j] << std::endl;
					std::cout << "m=" << m << std::endl;
				}
			} catch (exception &p) {
				continue;
			}
		}
	}
	Mat s_image;
	s_image.create(rows, cols, CV_16UC1);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {

			double d_val = sqrt(getter(0, sMatrix, i, j));
			int val = d_val * 255;
			s_image.at<unsigned short>(i, j) = val;
		}
	}
	std::cout << "writing S" << std::endl;
	imwrite("pcds/S.jpeg", s_image);
	std::cout << "wrote S" << std::endl;

	s_file.close();
	u_file.close();
	h_file.close();
	m_file.close();
	pq_file.close();
}

void SFS::read_skin_from_file(String filename, Point6dMatrix &m) {
	std::cout << "read_skin_from_file " << filename << std::endl;
	std::ifstream s_file(filename);
	std::string line;
	while (std::getline(s_file, line)) {
		std::istringstream iss(line);
		int i, j;
		double sc0, sc1, sc2, sc3, sc4, sc5;

		if (iss >> i >> j >> sc0 >> sc1 >> sc2 >> sc3 >> sc4 >> sc5) {
			markSelectedPixel(i, j);
			m[i][j] = Point6d(sc0, sc1, sc2, sc3, sc4, sc5);
			if (isSelectedPixel(i, j)) {
				std::cout << "data=" << m[i][j] << std::endl;
			}

		} else {
			return;
		}
	}
}

void SFS::read_skin_from_files() {
	std::cout << "read_skin_from_files" << std::endl;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			sMatrix[i][j] = { };
			uTestMatrix[i][j] = { };
			mTestMatrix[i][j] = { };
			hTestMatrix[i][j] = { };
			pqMatrix[i][j] = { };
		}
	}
	read_skin_from_file(skin + "/s.txt", sMatrix);
	read_skin_from_file(skin + "/u.txt", uTestMatrix);
	read_skin_from_file(skin + "/h.txt", hTestMatrix);
	read_skin_from_file(skin + "/m.txt", mTestMatrix);
	read_skin_from_file(skin + "/pq.txt", pqMatrix);
}

void SFS::print_test(int ci, int cj, int di, int dj) {

	std::cout << "================ print_test =====================" << std::endl;

	std::cout << "ci=" << ci << " cj=" << cj << std::endl;
	std::cout << "di=" << di << " dj=" << dj << std::endl;

	Point6d s0 = sMatrix[ci][cj].value();
	Point6d u0 = uTestMatrix[ci][cj].value();
	Point6d m0 = mTestMatrix[ci][cj].value();

	Point6d s = sMatrix[di][dj].value();
	Point6d u = uTestMatrix[di][dj].value();
	Point6d m = mTestMatrix[di][dj].value();

	std::cout << "s0=" << s0 << std::endl;
	std::cout << "u0=" << u0 << std::endl;
	std::cout << "m0=" << m0 << std::endl;

	std::cout << "s=" << s << std::endl;
	std::cout << "u=" << u << std::endl;
	std::cout << "m=" << m << std::endl;

	std::cout << "================ A =====================" << std::endl;

	double cx = scaleX(cj);
	double cy = scaleY(ci);
	double xx = scaleX(dj);
	double yy = scaleY(di);
	double dx = xx - cx;
	double dy = yy - cy;

	gsl_matrix *tau = getTau(dx, dy);
	print_matrix("tau=", tau);
	gsl_vector *m0_vec = m0.to_gsl_vector();
	matrix_times_vector(tau, m0_vec);
	print_vec("tau * m0=", m0_vec);

	gsl_matrix *pi0 = getA(s0, 0, 0);
	gsl_vector *u0_vec = u0.to_gsl_vector();
	matrix_times_vector(pi0, u0_vec);
	print_vec("pi0 * u0=", u0_vec);
	gsl_matrix_free(pi0);
	gsl_vector_free(u0_vec);

	gsl_matrix *pi = getA(s, dx, dy);
	u0_vec = u0.to_gsl_vector();
	matrix_times_vector(pi, u0_vec);
	print_vec("pi * u0=", u0_vec);
	gsl_matrix_free(pi);
	gsl_vector_free(u0_vec);
}

void SFS::generate_all_skin_coefficients() {
	markStepAlways("generate_all_skin_coefficients");
	std::cout << "si=" << si << " sj=" << sj << " testcoeff1=" << testcoeff1 << " testcoeff2=" << testcoeff2 << " testcoeff3=" << testcoeff3 << std::endl;
	double sX = scaleX(sj);
	double sY = scaleY(si);
	auto zFunction = [this, sX, sY](GiNaC::symbol x, GiNaC::symbol y) {
		if (test == 0) {
			return testcoeff1 * pow(x - sX, 2) + testcoeff2 * pow(y - sY, 2) + testcoeff3 * (x - sX) * (y - sY) + testcoeff4 * (x - sX) + testcoeff5 * (y - sY) + testcoeff6;
		} else if (test == 1) {
			GiNaC::ex r2 = sqr(testcoeff1);
			return GiNaC::sqrt(r2 - testcoeff2 * GiNaC::pow(x - sX, 2) - testcoeff3 * GiNaC::pow(y - sY, 2));
		} else if (test == 2) {
			GiNaC::ex r2 = sqr(testcoeff1);
			return GiNaC::sqrt(r2 - testcoeff2 * GiNaC::pow(x - sX, 2) - testcoeff3 * GiNaC::pow(y - sY, 2));
		} else if (test == 3) {
			return x * GiNaC::exp(-GiNaC::pow(x - sX, 2) - GiNaC::pow(y - sY, 2));
		}
		return GiNaC::pow(x - sX, 1);
	};

	auto zDomain = [this, sX, sY](double x, double y) {
		if (test == 0) {
			//return abs(x) < 0.3 && abs(y) < 0.3;
			return true;
		} else if (test == 1) {
			double z2 = sqr(testcoeff1) - testcoeff2 * pow(x - sX, 2) - testcoeff3 * pow(y - sY, 2);
			return z2 >= 0;
		} else if (test == 2) {
			return true;
		} else if (test == 3) {
			return true;
		}
		return false;
	};

	calculate_skin_test(zFunction, zDomain);
}

void SFS::generate_skin() {

	markStepAlways("generate_skin");
	sMatrix.resize(boost::extents[rows][cols]);
	uTestMatrix.resize(boost::extents[rows][cols]);
	hTestMatrix.resize(boost::extents[rows][cols]);
	mTestMatrix.resize(boost::extents[rows][cols]);
	normals_expected.resize(boost::extents[rows][cols]);
	pqMatrix.resize(boost::extents[rows][cols]);
	if (generated_derivatives == 1 || generated_derivatives == 2) {
		generate_all_skin_coefficients();
	}

	if (generated_derivatives == 0 || generated_derivatives == 3) {
		read_skin_from_files();
	}

	if (generated_derivatives == 0 || generated_derivatives == 2) {
		recover_all_skin_coefficients();
	}

	std::cout << "selected_i=" << selected_i << " selected_j=" << selected_j << std::endl;

	if (!sMatrix[di][dj]) {
		return;
	}

	Point6d s = sMatrix[di][dj].value();
	Point6d u = uTestMatrix[di][dj].value();
	Point6d m = mTestMatrix[di][dj].value();

	markStepAlways("print useful values");
	std::cout << "s=" << s << std::endl;
	gsl_matrix *pi = getA(s, scaleX(dj), scaleY(di));
	//gsl_matrix * pi = getA(s, 0, 0);
	print_matrix("pi", pi);
	std::cout << "u=" << u << std::endl;
	gsl_vector *u_vec = u.to_gsl_vector();
	matrix_times_vector(pi, u_vec);
	print_vec("pi * u", u_vec);
	std::cout << "m=" << m << std::endl;
}

void SFS::collect_neighbors(vector<std::pair<int, int>> &pixels) {
	for (int kk = 0; kk < rows; kk++) {
		for (int mm = 0; mm < cols; mm++) {
			if (getPixelValue(kk, mm) > 0) {
				pixels.push_back(std::pair<int, int>(mm, kk));
			}
		}
	}
}

double SFS::getGridValue() {
	int basek = rows / 2;
	int basem = cols / 2;
	if (rows % 2 == 0) {
		double val0 = getPixelValue(basek, basem);
		double val1 = getPixelValue(basek, basem + 1);
		double val2 = getPixelValue(basek + 1, basem);
		double val3 = getPixelValue(basek + 1, basem + 1);
		return (val0 + val1 + val2 + val3) / 4;
	} else {
		return getPixelValue(basek + 1, basem + 1);
	}
}

Point14d SFS::recover_skin_by_polynomial_fit() {

	vector<std::pair<int, int>> pixels;
	collect_neighbors(pixels);

//        for (int k = 0; k < pixels.size(); k++) {
//             std::cout << " pix=(" << pixels[k].second << ", " << pixels[k].first << ")" << std::endl;
//        }

	int numvars = 20;
	int sz = pixels.size();
	gsl_matrix *A = gsl_matrix_alloc(sz, numvars);
	gsl_matrix *Acopy = gsl_matrix_alloc(sz, numvars);
	gsl_vector *b = gsl_vector_alloc(sz);

	//std::cout << "1" << std::endl;

	double s0 = getGridValue();
	for (int kk = 0; kk < sz; kk++) {
		//std::cout << "2" << std::endl;
		if (fillSkinMatrixRow(kk, pixels[kk], s0, cx, cy, A, b)) {
			continue;
		}
	}
	//std::cout << "3" << std::endl;
	gsl_matrix_memcpy(Acopy, A);

	gsl_matrix *V = gsl_matrix_alloc(numvars, numvars);
	gsl_vector *S = gsl_vector_alloc(numvars);
	gsl_vector *work = gsl_vector_alloc(numvars);
	gsl_linalg_SV_decomp(A, V, S, work);

	gsl_vector *X = gsl_vector_alloc(numvars);
	// std::cout << "4" << std::endl;
	gsl_linalg_SV_solve(A, V, S, b, X);

	int row = 0;
	double s1 = gsl_vector_get(X, row++);
	double s2 = gsl_vector_get(X, row++);
	double s3 = gsl_vector_get(X, row++);
	double s4 = gsl_vector_get(X, row++);
	double s5 = gsl_vector_get(X, row++);
	double s6 = gsl_vector_get(X, row++);
	double s7 = gsl_vector_get(X, row++);
	double s8 = gsl_vector_get(X, row++);
	double s9 = gsl_vector_get(X, row++);
	double s10 = gsl_vector_get(X, row++);
	double s11 = gsl_vector_get(X, row++);
	double s12 = gsl_vector_get(X, row++);
	double s13 = gsl_vector_get(X, row++);
	double s14 = gsl_vector_get(X, row++);

	Point14d s(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14);

	if (isSelectedPixel()) {
		std::cout << " s=" << s << std::endl;
	}

	gsl_matrix_free(A);
	gsl_matrix_free(Acopy);
	gsl_matrix_free(V);
	gsl_vector_free(b);
	gsl_vector_free(S);
	gsl_vector_free(work);
	gsl_vector_free(X);

	return s;
}

Point6d SFS::calculate_skin_coefficients(Point14d &ss) {

	int i = selected_i;
	int j = selected_j;
	double x = scaleX(j);
	double y = scaleY(i);

	double dx = x - cx;
	double dy = y - cy;
	double sc0 = ss.eval(dx, dy);
	double sc1 = ss.evalDx(dx, dy);
	double sc2 = ss.evalDy(dx, dy);
	double sc3 = ss.evalDxx(dx, dy) / 2;
	double sc4 = ss.evalDxy(dx, dy);
	double sc5 = ss.evalDyy(dx, dy) / 2;

	Point6d sc(sc0, sc1, sc2, sc3, sc4, sc5);

	return sc;
}

bool SFS::fillSkinMatrixRow(int k, std::pair<int, int> &p, double s0, double cx, double cy, gsl_matrix *A, gsl_vector *b) {
	int i = p.second;
	int j = p.first;
	double X = scaleX(j) - cx;
	double Y = scaleY(i) - cy;
	double X2 = sqr(X);
	double X3 = X * X2;
	double X4 = X2 * X2;
	double Y2 = sqr(Y);
	double Y3 = Y * Y2;
	double Y4 = Y2 * Y2;

	gsl_vector_set(b, k, getPixelValue(i, j) - s0);
	int col = 0;
	//gsl_matrix_set(A, k, col++, 1);
	gsl_matrix_set(A, k, col++, X);
	gsl_matrix_set(A, k, col++, Y);

	gsl_matrix_set(A, k, col++, X2);
	gsl_matrix_set(A, k, col++, X * Y);
	gsl_matrix_set(A, k, col++, Y2);

	gsl_matrix_set(A, k, col++, X3);
	gsl_matrix_set(A, k, col++, X2 * Y);
	gsl_matrix_set(A, k, col++, X * Y2);
	gsl_matrix_set(A, k, col++, Y3);

	gsl_matrix_set(A, k, col++, X2 * X2);
	gsl_matrix_set(A, k, col++, X2 * X * Y);
	gsl_matrix_set(A, k, col++, X2 * Y2);
	gsl_matrix_set(A, k, col++, X * Y2 * Y);
	gsl_matrix_set(A, k, col++, Y2 * Y2);

	gsl_matrix_set(A, k, col++, X4 * X);
	gsl_matrix_set(A, k, col++, X4 * Y);
	gsl_matrix_set(A, k, col++, X3 * Y2);
	gsl_matrix_set(A, k, col++, X2 * Y3);
	gsl_matrix_set(A, k, col++, X * Y4);
	gsl_matrix_set(A, k, col++, Y4 * Y);
	return true;
}

void SFS::setMatrixRow(int k, Point6d vc, gsl_matrix *A) {
	int col = 0;
	gsl_matrix_set(A, k, col++, vc.x0);
	gsl_matrix_set(A, k, col++, vc.x1);
	gsl_matrix_set(A, k, col++, vc.x2);
	gsl_matrix_set(A, k, col++, vc.x3);
	gsl_matrix_set(A, k, col++, vc.x4);
	gsl_matrix_set(A, k, col++, vc.x5);

	if (computeLight == 1) {
		gsl_matrix_set(A, k, col++, -1);
	}
}

gsl_matrix* SFS::getPi(Point6d &c) {
	gsl_matrix *pi = gsl_matrix_alloc(6, 6);
	gsl_matrix_set_all(pi, 0.0);
	for (int k = 0; k < 6; k++) {
		gsl_matrix_set(pi, k, k, c.x0);
		gsl_matrix_set(pi, k, 0, c.get(k));
	}
	gsl_matrix_set(pi, 3, 1, c.x1);
	gsl_matrix_set(pi, 3, 3, c.x0);
	gsl_matrix_set(pi, 4, 1, c.x2);
	gsl_matrix_set(pi, 4, 2, c.x1);
	gsl_matrix_set(pi, 5, 2, c.x2);

	return pi;
}

gsl_matrix* SFS::getTau(double x, double y) {
	gsl_matrix *tau = gsl_matrix_alloc(6, 6);
	gsl_matrix_set_all(tau, 0.0);
	for (int k = 0; k < 6; k++) {
		gsl_matrix_set(tau, k, k, 1.0);
	}
	gsl_matrix_set(tau, 0, 1, x);
	gsl_matrix_set(tau, 0, 2, y);
	gsl_matrix_set(tau, 0, 3, sqr(x));
	gsl_matrix_set(tau, 0, 4, x * y);
	gsl_matrix_set(tau, 0, 5, sqr(y));

	gsl_matrix_set(tau, 1, 3, 2 * x);
	gsl_matrix_set(tau, 1, 4, y);

	gsl_matrix_set(tau, 2, 4, x);
	gsl_matrix_set(tau, 2, 5, 2 * y);

	return tau;
}

gsl_matrix* SFS::getA(Point6d &sc, double x, double y) {

	gsl_matrix *pi = getPi(sc);
	gsl_matrix *tau = getTau(x, y);
	gsl_matrix *pp = matrix_times_matrix(pi, tau);
	gsl_matrix_free(pi);
	gsl_matrix_free(tau);
	return pp;
}

void SFS::collect_neighborhood_pixels(vector<std::pair<int, int>> &pixels) {
	int i = selected_i;
	int j = selected_j;
	int w = w_coeffs;
	int w2 = 0.7 * w;

	pixels.push_back(std::pair<int, int>(j - w2, i - w2));
	pixels.push_back(std::pair<int, int>(j - w2, i + w2));
	pixels.push_back(std::pair<int, int>(j + w2, i - w2));
	pixels.push_back(std::pair<int, int>(j + w2, i + w2));

	pixels.push_back(std::pair<int, int>(j, i - w));
	pixels.push_back(std::pair<int, int>(j, i + w));
	pixels.push_back(std::pair<int, int>(j - w, i));
	pixels.push_back(std::pair<int, int>(j + w, i));
}

boost::optional<Point6d> SFS::getU(gsl_matrix *V, gsl_vector *S, int numvars) {

	for (int k = lastCol; k >= 0; k--) {
		int row = 0;
		double u0 = gsl_matrix_get0(V, row++, k);
		double u1 = gsl_matrix_get0(V, row++, k);
		double u2 = gsl_matrix_get0(V, row++, k);
		double u3 = gsl_matrix_get0(V, row++, k);
		double u4 = gsl_matrix_get0(V, row++, k);
		double u5 = gsl_matrix_get0(V, row++, k);

		// both u0 and u1 must be positive but
		// because they are a solution of a homogenous system
		// it could be that they are negative, so...

		if (u0 == 0) {
			continue;
		}

		Point6d u(u0, u1, u2, u3, u4, u5);
		if (u3 < 0 || u5 < 0) {
			u = u.scale(-1);
		}

		if (u.x3 * u.x5 < 0) {
			continue;
		}
		return u;            //.getU0(getX(), getY());
	}
	return {};
}

boost::optional<Point6d> SFS::getM(gsl_matrix *V, int numvars) {

	int row = 6;

	double m0 = gsl_matrix_get0(V, row++, lastCol);
	double m1 = gsl_matrix_get0(V, row++, lastCol);
	double m2 = gsl_matrix_get0(V, row++, lastCol);
	double m3 = gsl_matrix_get0(V, row++, lastCol);
	double m4 = gsl_matrix_get0(V, row++, lastCol);
	double m5 = gsl_matrix_get0(V, row++, lastCol);

	if (m3 * m5 < 0) {
		return {};
	}

	Point6d m(m0, m1, m2, m3, m4, m5);
	if (m3 < 0 || m5 < 0) {
		m = m.scale(-1);
	}
	return m;
}

bool SFS::fillNormalMatrixRow(int k, std::pair<int, int> &p, gsl_matrix *M) {

	int i = p.second;
	int j = p.first;
	boost::optional<Point6d> s0 = sMatrix[selected_i][selected_j];
	boost::optional<Point6d> sc = sMatrix[i][j];
	if (!s0 || !sc) {
		return false;
	}

	double x = getX();
	double y = getY();

	double xx = scaleX(j);
	double yy = scaleY(i);

	double dx = xx - x;
	double dy = yy - y;

	gsl_matrix *pi = getA(sc.value(), dx, dy);
	gsl_matrix *tau = getTau(dx, dy);
	for (int n = 0; n < 6; n++, k++) {
		for (int m = 0; m < 6; m++) {
			gsl_matrix_set(M, k, m, gsl_matrix_get(pi, n, m));
			gsl_matrix_set(M, k, m + 6, -gsl_matrix_get(tau, n, m));
		}
	}

	if (isSelectedPixel()) {
		print_test(selected_i, selected_j, i, j);
	}
	gsl_matrix_free(pi);
	gsl_matrix_free(tau);
	return true;
}

Point3d SFS::toLight(Point6d &h, int i, int j) {

	double y = scaleY(i);
	double x = scaleX(j);

	int lc = llw * 0.707;
	vector<std::pair<int, int>> pixels;
	pixels.push_back(std::pair<int, int>(j + lc, i + lc));
	pixels.push_back(std::pair<int, int>(j + lc, i - lc));
	pixels.push_back(std::pair<int, int>(j - lc, i + lc));
	pixels.push_back(std::pair<int, int>(j - lc, i - lc));
	pixels.push_back(std::pair<int, int>(j + llw, i));
	pixels.push_back(std::pair<int, int>(j - llw, i));
	pixels.push_back(std::pair<int, int>(j, i + llw));
	pixels.push_back(std::pair<int, int>(j, i - llw));

	int numvars = 3;
	int sz = pixels.size();
	std::cout << "light pixel count=" << sz << std::endl;
	gsl_matrix *A = gsl_matrix_alloc(sz, numvars);
	gsl_vector *b = gsl_vector_alloc(sz);
	gsl_matrix_set_all(A, 0);
	for (int k = 0; k < sz; k++) {
		std::pair<int, int> pp = pixels[k];
		int ii = pp.second;
		int jj = pp.first;
		double dy = scaleY(ii) - y;
		double dx = scaleX(jj) - x;
		double p = 2 * h.x3 * dx + h.x4 * dy + h.x1;
		double q = 2 * h.x5 * dy + h.x4 * dx + h.x2;
		double norm = sqrt(sqr(p) + sqr(q) + 1);
		//std::cout << "p=" << p << " q=" << q << " norm=" << norm << std::endl;

		gsl_matrix_set(A, k, 0, -p / norm);
		gsl_matrix_set(A, k, 1, -q / norm);
		gsl_matrix_set(A, k, 2, 1 / norm);
		gsl_vector_set(b, k, sqrt(getPixelValue(ii, jj)));
	}

	print_matrix("A", A, true);
	print_vec("b", b, true);

	gsl_matrix *V = gsl_matrix_alloc(3, 3);
	gsl_vector *S = gsl_vector_alloc(3);
	gsl_vector *work = gsl_vector_alloc(3);
	gsl_linalg_SV_decomp(A, V, S, work);

	gsl_vector *X = gsl_vector_alloc(3);
	gsl_linalg_SV_solve(A, V, S, b, X);

	double lx = normalize0(gsl_vector_get(X, 0));
	double ly = normalize0(gsl_vector_get(X, 1));
	double lz = normalize0(gsl_vector_get(X, 2));

	Point3d l = unit(Point3d(lx, ly, lz));

	gsl_matrix_free(A);
	gsl_matrix_free(V);
	gsl_vector_free(S);
	gsl_vector_free(work);
	gsl_vector_free(X);

	return l;
}

void SFS::addSolution(Point6d &u, double h3, double h4, double h5, vector<Point6d> &solutions) {

	double h4_squared = sqr(h4);
	double detH = normalize0(4 * h3 * h5 - h4_squared);

	if (isSelectedPixel()) {
		std::cout << "h3=" << h3 << " h5=" << h5 << " h4=" << h4 << " h4_squared=" << h4_squared << " detH=" << detH << std::endl;
	}

	if (isSelectedPixel()) {
		std::cout << "before det check=" << std::endl;
	}
	if (is0(detH)) {
		if (isSelectedPixel()) {
			std::cout << "is 0=" << std::endl;
		}
		return;
	}
	double u0 = u.x0;
	double u1 = u.x1;
	double u2 = u.x2;
	double u3 = u.x3;
	double u4 = u.x4;
	double u5 = u.x5;

	double detHInv = 1 / detH;
	double p = detHInv * (h3 * u1 - h4 * u2 / 2);
	double q = detHInv * (-h4 * u1 / 2 + h3 * u2);
	if (isSelectedPixel()) {
		std::cout << "p before = " << p << " q before =" << q << std::endl;
	}
	double X = getX();
	double Y = getY();

	double h1 = p;           	// - 2 * h3 * X - h4 * Y;
	double h2 = q;           	// - 2 * h3 * Y - h4 * X;

	// calculate homogeneity factor
	double s_inv = u0 - (u3 * sqr(u1) - u4 * u1 * u2 + u5 * sqr(u2)) * 0.25 * sqr(detHInv);
	//double s_inv = u0 - sqr(h1) - sqr(h2);
	//   std::cout << "s_inv=" << s_inv << std::endl;

	if (isSelectedPixel()) {
		std::cout << "s_inv=" << s_inv << std::endl;
	}

	if (s_inv < 0) {
		return;
	}

	double s = 1 / s_inv;

	Point6d us = u.scale(s);
	if (isSelectedPixel()) {
		std::cout << "u scaled=" << us << std::endl;
		std::cout << "u      0=" << us.getU0(getX(), getY()) << std::endl;
		std::cout << "u      1=" << us.getU1(getX(), getY()) << std::endl;
		;
	}
	double s_root = sqrt(s);
	if (isSelectedPixel()) {
		std::cout << "s_root=" << s_root << std::endl;
	}

	Point6d h_unscaled(0, h1, h2, h3, h4, h5);
	Point6d h = h_unscaled.scale(s_root);

	solutions.push_back(h);
}

void SFS::addSolutions(Point6d &u, double d_root, vector<Point6d> &solutions) {

	double u0 = u.x0;
	double u1 = u.x1;
	double u2 = u.x2;
	double u3 = u.x3;
	double u4 = u.x4;
	double u5 = u.x5;

	double h4_squared = 0;
	if (u4 != 0) {
		h4_squared = normalize0(0.25 * (u3 + u5 + d_root) / (1 + sqr((u3 - u5) / u4)));
		if (isSelectedPixel()) {
			std::cout << " h4_squared=" << h4_squared << std::endl;
		}
		if (h4_squared < 0 || u3 < h4_squared || u5 < h4_squared) {
			return;
		}
	}

	double h3 = sqrt(max(u3 - h4_squared, 0.0)) / 2;
	double h4 = sqrt(h4_squared);
	double h5 = sqrt(max(u5 - h4_squared, 0.0)) / 2;

	if (is0(u4, epsilon_m0)) {
		addSolution(u, h3, 0, h5, solutions);
		addSolution(u, h3, 0, -h5, solutions);
		addSolution(u, -h3, 0, h5, solutions);
		addSolution(u, -h3, 0, -h5, solutions);
	} else {

		double delta0 = normalize0(abs(abs(u4) - 4 * h4 * (h3 + h5)));
		double delta1 = normalize0(abs(abs(u4) - 4 * h4 * abs(h3 - h5)));
		if (isSelectedPixel()) {
			std::cout << "delta0=" << delta0 << " delta1=" << delta1 << std::endl;
		}
		//double hh4 = ((delta0 <= delta1) ? 1 : sgn(h3 - h5)) * sgn(u4) * h4;
		h4 *= sgn(u4);
		if (delta0 > delta1) {
			if (h3 < h5) {
				h3 *= -1;
			} else {
				h5 *= -1;
			}
		}

		addSolution(u, h3, h4, h5, solutions);
		addSolution(u, -h3, -h4, -h5, solutions);
	}
}

void SFS::calculate_surface_coefficients() {

	int i = selected_i;
	int j = selected_j;

	boost::optional<Point6d> u = uMatrix[i][j];
	boost::optional<Point6d> s = sMatrix[i][j];
	if (!u || !s) {
		return;
	}

	Point6d uu = u.value();

	double u3 = uu.x3;
	double u4 = uu.x4;
	double u5 = uu.x5;
	double u4_squared = sqr(u4);

	double discriminant = normalize0(4 * u3 * u5 - u4_squared);

	if (isSelectedPixel()) {
		std::cout << "u4=" << u4 << " u4_squared=" << u4_squared << " 4 * u3 * u5=" << (4 * u3 * u5) << " u4_squared / (4 * u3 * u5)=" << (u4_squared / (4 * u3 * u5)) << std::endl;
		std::cout << "discriminant=" << discriminant << std::endl;
	}
	if (discriminant < 0) {
		return;
	}

	vector<Point6d> solutions;
	double d_root = sqrt(discriminant);
	addSolutions(uu, d_root, solutions);
	if (u4 != 0 && discriminant > 0) {
		addSolutions(uu, -d_root, solutions);
	}
	// there are alwys either 4 or 0 solutions

	if (solutions.size() > 0) {
		zMatrix0[i][j] = solutions[0];
		zMatrix1[i][j] = solutions[1];
		zMatrix2[i][j] = solutions[2];
		zMatrix3[i][j] = solutions[3];
	}
}

boost::optional<Point6d> SFS::calculate_normal_coefficients() {

	int numvars = 12;
	vector<std::pair<int, int>> pixels;
	collect_neighborhood_pixels(pixels);
	int sz = pixels.size() * 6;
	gsl_matrix *M = gsl_matrix_alloc(sz, numvars);
	gsl_matrix_set_all(M, 0);

	for (int k = 0; k < pixels.size(); k++) {
		if (isSelectedPixel()) {
			std::cout << "pix=(" << pixels[k].first << ", " << pixels[k].second << ")" << std::endl;
		}
		if (!fillNormalMatrixRow(k * 6, pixels[k], M)) {
			return {};
		}
	}
	gsl_matrix *Mcopy = gsl_matrix_alloc(sz, numvars);
	gsl_matrix_memcpy(Mcopy, M);
	if (isSelectedPixel()) {
		print_matrix("M", Mcopy);
	}

	gsl_matrix *X = gsl_matrix_alloc(numvars, numvars);
	gsl_matrix *V = gsl_matrix_alloc(numvars, numvars);
	gsl_vector *S = gsl_vector_alloc(numvars);
	gsl_vector *work = gsl_vector_alloc(numvars);

	gsl_matrix_set_all(V, 0);
	gsl_vector_set_all(S, 0);
	gsl_vector_set_all(work, 0);

	gsl_linalg_SV_decomp(M, V, S, work);
	//gsl_linalg_SV_decomp_mod(M, X, V, S, work);
	if (isSelectedPixel()) {
		print_matrix("U", M);
		//  print_matrix("X", X);
		print_matrix("V", V);
		print_vec("S", S);
	}

	boost::optional<Point6d> u = getU(V, S, numvars);
	boost::optional<Point6d> m = getM(V, numvars);

	uMatrix[selected_i][selected_j] = u;
	mMatrix[selected_i][selected_j] = m;

	gsl_matrix_free(M);
	gsl_matrix_free(Mcopy);
	gsl_matrix_free(X);
	gsl_matrix_free(V);
	gsl_vector_free(S);
	gsl_vector_free(work);

	if (isSelectedPixel()) {
		std::cout << "u=" << u << std::endl;
	}
	return u;
}

void SFS::recover_all_skin_coefficients() {

	Point14d s = recover_skin_by_polynomial_fit();
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			markSelectedPixel(i, j);
			sMatrix[i][j] = calculate_skin_coefficients(s);
		}
	}
}

void SFS::recover_all_normal_coefficients() {
	uMatrix.resize(boost::extents[rows][cols]);
	mMatrix.resize(boost::extents[rows][cols]);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			uMatrix[i][j] = { };
			mMatrix[i][j] = { };
			markSelectedPixel(i, j);
			if (isBorderPixel()) {
				continue;
			}
			calculate_normal_coefficients();
		}
	}
}

void SFS::recover_all_surface_coefficients() {

	double sX = scaleX(sj);
	double sY = scaleY(si);

	auto zDomain = [this, sX, sY](double x, double y) {
		if (test == 0) {
			//return abs(x) < 0.3 && abs(y) < 0.3;
			return true;
		} else if (test == 1) {
			double z2 = sqr(testcoeff1) - testcoeff2 * pow(x - sX, 2) - testcoeff3 * pow(y - sY, 2);
			return z2 > 0;
		} else if (test == 2) {
			return true;
		} else if (test == 3) {
			return true;
		}
		return false;
	};

	zMatrix0.resize(boost::extents[rows][cols]);
	zMatrix1.resize(boost::extents[rows][cols]);
	zMatrix2.resize(boost::extents[rows][cols]);
	zMatrix3.resize(boost::extents[rows][cols]);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			markSelectedPixel(i, j);

			if (isSelectedPixel(i, j)) {
				std::cout << "try calculate_surface_coefficients" << std::endl;
			}
			zMatrix0[i][j] = { };
			zMatrix1[i][j] = { };
			zMatrix2[i][j] = { };
			zMatrix3[i][j] = { };

			if (isBorderPixel() || !zDomain(getX(), getY())) {
				continue;
			}
			calculate_surface_coefficients();
		}
	}

}

void SFS::allocate_height_matrices(int sz, gsl_matrix **A, gsl_matrix **Acopy, gsl_matrix **V, gsl_vector **b, gsl_vector **S, gsl_vector **work, gsl_vector **X) {
	*A = gsl_matrix_alloc(sz, sz);
	*Acopy = gsl_matrix_alloc(sz, sz);
	*b = gsl_vector_alloc(sz);
	*V = gsl_matrix_alloc(sz, sz);
	*S = gsl_vector_alloc(sz);
	*work = gsl_vector_alloc(sz);
	*X = gsl_vector_alloc(sz);

	gsl_matrix_set_all(*A, 0);
	gsl_matrix_set_all(*Acopy, 0);
	gsl_vector_set_all(*b, 0);
	gsl_matrix_set_all(*V, 0);
	gsl_vector_set_all(*S, 0);
	gsl_vector_set_all(*work, 0);
	gsl_vector_set_all(*X, 0);

}

void SFS::print_height_matrices(gsl_matrix *A, gsl_matrix *Acopy, gsl_matrix *V, gsl_vector *b, gsl_vector *S, gsl_vector *work, gsl_vector *X) {
	print_matrix("A", Acopy);
	print_matrix("U", A);
	print_matrix("V", V);
	print_vec("S", S);
	print_vec("w", work);
	print_vec("b", b);
	print_vec("X", X);

}

void SFS::free_height_matrices(gsl_matrix *A, gsl_matrix *Acopy, gsl_matrix *V, gsl_vector *b, gsl_vector *S, gsl_vector *work, gsl_vector *X) {
	gsl_matrix_free(A);
	gsl_matrix_free(Acopy);
	gsl_matrix_free(V);
	gsl_vector_free(b);
	gsl_vector_free(S);
	gsl_vector_free(work);
	gsl_vector_free(X);
}

void SFS::balance_heights(int h, int w, std::function<double(int, int, int, int)> getRHS, std::function<void(gsl_vector*)> transferHeights) {

	int sz = h * w - 1;
	int limi = h - 1;
	int limj = w - 1;

	gsl_matrix *A;
	gsl_matrix *Acopy;
	gsl_matrix *V;
	gsl_vector *b;
	gsl_vector *S;
	gsl_vector *work;
	gsl_vector *X;

	allocate_height_matrices(sz, &A, &Acopy, &V, &b, &S, &work, &X);

	//    std::cout << "limi=" << limi << " limj=" << limj << std::endl;
	for (int i = 0; i <= limi; i++) {
		for (int j = 0; j <= limj; j++) {

			if (i == limi && j == limj) {
				continue;
			}

			int si = i < limi;
			int sj = j < limj;

			int k = i * h + j;

			gsl_matrix_set(A, k, k, -(si + sj));

			if (sj > 0 && !(j == limj - 1 && i == limi)) {
				gsl_matrix_set(A, k, k + 1, sj);
			}

			if (si > 0 && !(i == limi - 1 && j == limj)) {
				gsl_matrix_set(A, k, k + w, si);
			}

			//    std::cout << "i=" << i << " j=" << j << " si=" << si << " sj=" << sj << std::endl;
			gsl_vector_set(b, k, getRHS(i, j, si, sj));
		}
	}
	gsl_matrix_memcpy(Acopy, A);

	gsl_linalg_SV_decomp(A, V, S, work);
	gsl_linalg_SV_solve(A, V, S, b, X);

	transferHeights(X);

	//    print_height_matrices(A, Acopy, V, b, S, work, X);
	free_height_matrices(A, Acopy, V, b, S, work, X);
}

void SFS::convert_normals_to_height_h0(int k, int m, int h, int w, Point6dMatrix &zMatrix, HeightMatrix &hMatrix) {

	int topi = k * h;
	int topj = m * w;

	double dx = getDeltaX();
	double dy = getDeltaY();

	auto getRHS = [this, &zMatrix, &hMatrix, topi, topj, dx, dy](int i, int j, int si, int sj) {

		int reali = topi + i;
		int realj = topj + j;

		markSelectedPixel(reali, realj);
		double p = (double) getter(pGetter, zMatrix, reali, realj);
		double q = (double) getter(qGetter, zMatrix, reali, realj);

		double rhs = sj * p * dx - si * q * dy;

		return rhs;
	};

	auto transferHeights = [this, &zMatrix, &hMatrix, topi, topj, h, w, dx](gsl_vector *height) {

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {

				double h0 = 0;

				int reali = topi + i;
				int realj = topj + j;

				if (i == h - 1 && j == w - 1) {
					if (hMatrix[reali][realj - 1]) {
						double p = (double) getter(pGetter, zMatrix, reali, realj - 1);
						h0 = p * dx + hMatrix[reali][realj - 1].value();
					}
				} else {
					h0 = gsl_vector_get(height, i * h + j);
				}

				hMatrix[reali][realj] = h0;
			}
		}
	};

	balance_heights(h, w, getRHS, transferHeights);
}

void SFS::convert_normals_to_height_h1(int k, int m, int h, int w, Point6dMatrix &zMatrix, HeightMatrix &hMatrix) {

	int realh = h * aspecth;
	int realw = w * aspectw;

	int topi = k * realh;
	int topj = m * realw;

	//std::cout << "convert_normals_to_height_h1 w=" << w << " h=" << h << " k=" << k << " m=" << m << " realh=" << realh << " realw=" << realw << " topi=" << topi << " topj=" << topj << std::endl;

	double dx = getDeltaX();
	double dy = getDeltaY();

	/*
	 Point6dMatrix zMat = zMatrix;
	 if (z0height == 1) {
	 zMat = hTestMatrix;
	 }
	 */
	auto getRHS = [this, &zMatrix, &hMatrix, dx, dy, topi, topj, k, m, h, w, realh, realw](int i, int j, int si, int sj) {

		int topii = topi + i * aspecth;
		int topjj = topj + j * aspectw;

		//std::cout << "i=" << i << " j=" << j << " getRHS topii=" << topii << " topjj=" << topjj << " h=" << h << " w=" << w << std::endl;

		double pp = 0;
		int fixedj = topjj + aspectw;
		if (sj > 0) {
			for (int ii = 0; ii < h; ii++) {
				int reali = topii + ii;
				//    std::cout << "reali=" << reali << " fixedj=" << fixedj << std::endl;
				markSelectedPixel(reali, fixedj);
				double p = (double) getter(pGetter, zMatrix, reali, fixedj - 1);
				double z = (double) getter(0, hMatrix, reali, fixedj - 1);
				double znext = (double) fixedj == cols ? z : getter(0, hMatrix, reali, fixedj);
				double delta = z + p * dx - znext;
				pp += delta;
			}
		}
		pp /= h;

		double qq = 0;
		if (si > 0) {
			int fixedi = topii + aspecth;
			for (int jj = 0; jj < w; jj++) {
				int realj = topjj + jj;
				markSelectedPixel(fixedi, realj);
				double q = (double) getter(qGetter, zMatrix, fixedi - 1, realj);
				double z = (double) getter(0, hMatrix, fixedi - 1, realj);
				double znext = (double) fixedi == rows ? z : getter(0, hMatrix, fixedi, realj);
				double delta = z - q * dy - znext;
				qq += delta;
			}
		}

		qq /= w;

		double rhs = si * qq + sj * pp;
		return rhs;
	};

	auto transferHeights = [this, &hMatrix, topi, topj, h, w](gsl_vector *height) {

		//    std::cout << "transferHeights topi=" << topi << " topj=" << topj << " h=" << h << " w=" << w << std::endl;
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {

				if (i == h - 1 && j == w - 1) {
					continue;
				}
				//        std::cout << "i=" << i << " j=" << j << std::endl;
				int idx = i * h + j;
				//        std::cout << "after idx=" << idx << " height size=" << height->size << std::endl;
				double h1 = gsl_vector_get(height, idx);
				// now add h1 to the heights of the previous iteration

				for (int ii = 0; ii < aspecth; ii++) {
					int reali = topi + i * aspecth + ii;
					for (int jj = 0; jj < aspectw; jj++) {
						int realj = topj + j * aspectw + jj;
						//    std::cout << "reali=" << reali << " realj=" << realj << std::endl;
						hMatrix[reali][realj].value() += h1;
					}
				}
			}
		}
	};

	balance_heights(h, w, getRHS, transferHeights);
}

void SFS::convert_normals_to_height_h0(Point6dMatrix &zMatrix, HeightMatrix &hMatrix) {
	for (int k = 0; k < 50; k++) {
		for (int m = 0; m < 50; m++) {
			convert_normals_to_height_h0(k, m, aspecth, aspectw, zMatrix, hMatrix);
		}
	}
	//convert_normals_to_height_h0(12, 12, aspecth, aspectw, zMatrix, hMatrix);
	//convert_normals_to_height_h0(38, 38, aspecth, aspectw, zMatrix, hMatrix);
}

void SFS::convert_normals_to_height_h1(Point6dMatrix &zMatrix, HeightMatrix &hMatrix) {
	for (int k = 0; k < 10; k++) {
		for (int m = 0; m < 10; m++) {
			convert_normals_to_height_h1(k, m, 5, 5, zMatrix, hMatrix);
		}
	}
	//convert_normals_to_height_h1(3, 3, 5, 5, zMatrix, hMatrix);
	//convert_normals_to_height_h1(6, 6, 5, 5, zMatrix, hMatrix);
	//convert_normals_to_height_h1(0, 0, 5, 5);
	//convert_normals_to_height_h1(0, 9, 5, 5);
	//convert_normals_to_height_h1(9, 0, 5, 5);
}

void SFS::convert_normals_to_height_h2(Point6dMatrix &zMatrix, HeightMatrix &hMatrix) {

	int h = 10;
	int w = 10;

	double dx = getDeltaX();
	double dy = getDeltaY();

	int blockheight = 5 * aspecth;
	int blockwidth = 5 * aspectw;

	/*
	 Point6dMatrix zMat = zMatrix;
	 if (z0height == 1) {
	 zMat = hTestMatrix;
	 }
	 */
	auto getRHS = [this, &zMatrix, &hMatrix, dx, dy, blockheight, blockwidth](int i, int j, int si, int sj) {

		int topi = i * blockheight;
		int topj = j * blockwidth;

		double pp = 0;
		int fixedj = topj + blockwidth;
		if (sj > 0) {
			for (int ii = 0; ii < blockheight; ii++) {
				int reali = topi + ii;
				//    std::cout << "reali=" << reali << " fixedj=" << fixedj << std::endl;
				markSelectedPixel(reali, fixedj);
				double p = (double) getter(pGetter, zMatrix, reali, fixedj - 1);
				double z = (double) getter(0, hMatrix, reali, fixedj - 1);
				double znext = (double) fixedj == cols ? z : getter(0, hMatrix, reali, fixedj);
				double delta = z + p * dx - znext;
				pp += delta;
			}
		}

		pp /= blockheight;

		double qq = 0;
		if (si > 0) {
			int fixedi = topi + blockheight;
			for (int jj = 0; jj < blockwidth; jj++) {
				int realj = topj + jj;
				markSelectedPixel(fixedi, realj);
				double q = (double) getter(qGetter, zMatrix, fixedi - 1, realj);
				double z = (double) getter(0, hMatrix, fixedi - 1, realj);
				double znext = (double) fixedi == rows ? z : getter(0, hMatrix, fixedi, realj);
				double delta = z - q * dy - znext;
				qq += delta;
			}
		}

		qq /= blockwidth;

		double rhs = si * qq + sj * pp;
		return rhs;
	};

	auto transferHeights = [this, &hMatrix, h, w, blockheight, blockwidth](gsl_vector *height) {

		for (int i = 0; i < h; i++) {
			int topi = i * blockheight;
			for (int j = 0; j < w; j++) {

				int topj = j * blockwidth;
				double h2 = gsl_vector_get(height, i * h + j);

				// now add h2 to the heights of the previous iteration

				for (int ii = 0; ii < blockheight; ii++) {
					int reali = topi + ii;
					for (int jj = 0; jj < blockwidth; jj++) {
						int realj = topj + jj;
						//    std::cout << "reali=" << reali << " realj=" << realj << std::endl;
						if (hMatrix[reali][realj]) {
							hMatrix[reali][realj].value() += h2;
						}
					}
				}
			}
		}
	};

	balance_heights(h, w, getRHS, transferHeights);
}

void SFS::convert_normals_to_height(Point6dMatrix &zMatrix, HeightMatrix &hMatrix) {
	hMatrix.resize(boost::extents[rows][cols]);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			hMatrix[i][j] = 0.0;
		}
	}
	convert_normals_to_height_h0(zMatrix, hMatrix);
	convert_normals_to_height_h1(zMatrix, hMatrix);
	convert_normals_to_height_h2(zMatrix, hMatrix);
}

void SFS::recover_shape() {

	markStepAlways("calculate_skin_coefficients");

	generate_skin();

	markStepAlways("recover_all_normal_coefficients");
	recover_all_normal_coefficients();
	markStepAlways("recover_all_surface_coefficients");
	recover_all_surface_coefficients();
	markStepAlways("convert_normals_to_height");

	clock_t begin = clock();
	convert_normals_to_height(zMatrix0, hMatrix0);
	convert_normals_to_height(zMatrix1, hMatrix1);
	convert_normals_to_height(zMatrix2, hMatrix2);
	convert_normals_to_height(zMatrix3, hMatrix3);
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "convert_normals_to_height took " << elapsed_secs << std::endl;

}
