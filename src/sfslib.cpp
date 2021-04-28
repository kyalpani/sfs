#include "sfslib.h"

double epsilon_zero;
int sel_i;
int sel_j;
const char *keys =
		{
				"{help h||}"
						"{@image                      |sfs.img | input image name}"
						"{skin                        | skin   | directory in which pre computed skin files for testing reside}"
						"{n_coeffs              count | 10     | number of bspline coefficients}"
						"{w_coeffs              count | 6      | window size for calculation normal coefficients}"
						"{generated_derivatives count | 0      | 0 if derivatives are generated 1 if they are computed from input data }"
						"{l1                          | 0      | x component of light vector [0..1)}"
						"{l2                          | 0      | y component of light vector [0..1)}"
						"{l3                          | 1      | y component of light vector (0..1]}"
						"{li                    count | 0      | the i-index of the image window for computing the light vector }"
						"{lj                    count | 0      | the j-index of the image window for computing the light vector }"
						"{sc                          |        | path to input sc file }"
						"{si                    count | 0      | the i-index of the sample point }"
						"{sj                    count | 0      | the j-index of the sample point }"
						"{lw                    count | 0      | the window size of the image window for computing the light vectors }"
				        "{llw                   count | 0      | the window size of the image window for computing the light vector }"
						"{di                    count | 0      | the i-index of the debug pixel }"
						"{dj                    count | 0      | the j-index of the debug pixel }"
						"{dw                    count | 0      | the window size of debug pixel window }"
						"{test                  count | 0      | the test number to use for generated derivates }"
						"{testcoeff1                  | 1      | the coefficient for test 1 }"
						"{testcoeff2                  | 1      | the coefficient for test 2 }"
						"{testcoeff3                  | 1      | the coefficient for test 3 }"
						"{testcoeff4                  | 0      | the coefficient for test 4 }"
						"{testcoeff5                  | 0      | the coefficient for test 5 }"
						"{testcoeff6                  | 0      | the coefficient for test 6 }"
						"{epsilon                     | 0.0001 | epsilon: error margin}"
						"{epsilon_m0                  | 0.01   | epsilon m0: error margin}"
						"{cuti                  count | -1     | cut the output along a fixed i}"
						"{cutj                  count | -1     | cut the output along a fixed j}"
						"{blocksize             count | 8      | blocksize for height recovery}"
						"{z0height              count | 8      | set to 1 if z0 should be used as basis for height calculation}"
						"{pGetter0              count | 12      | getter type for P0 values}"
						"{qGetter0              count | 13     | getter type for Q0 values}"
						"{pGetter               count | 12      | getter type for P values}"
						"{qGetter               count | 13     | getter type for Q values}"
						"{computeLight          count | 1      | when 0 light vector must be provided}"
						"{getter_upper_limit    count | 20     | cutoff limit for displaying cloud point heights}"
						"{lastCol               count | 8      | last column to pick out the solution from }"
						"{detLimitHigh                | 0.1    | determinant lower limit }"
						"{detLimitLow                 | 0.1    | determinant higher limit }"

		};

double sqr(double x) {
	return x * x;
}

double len(Point3d &a) {
	return sqrt(sqr(a.x) + sqr(a.y) + sqr(a.z));
}

Point3d unit(Point3d p) {
	double l = len(p);
	return Point3d(p.x / l, p.y / l, p.z / l);
}

Point2i toHemisphericIndex(Point3d l) {
	double norm = sqrt(sqr(l.x) + sqr(l.y));
	double altitude = atan2(l.z, norm) * 180 / PI;
	double azimuth = atan2(l.y, l.x) * 180 / PI;
	if (azimuth < 0) {
		azimuth = 2 * PI - azimuth;
	}

	Point2i hi(trunc(azimuth), trunc(altitude));
	return hi;
}

Point3d fromHemisphericIndex(Point2i h) {
	double azimuthRad = h.x * PI / 180;
	double altitudeRad = h.y * PI / 180;

	double norm = cos(altitudeRad);
	return Point3d(cos(azimuthRad) * norm, sin(azimuthRad) * norm, 1);
}

bool is0(double num, double epsilon) {
	return abs(num) < epsilon;
}

bool is0(double num) {
	return is0(num, epsilon_zero);
}

double normalize0(double x) {
	return is0(x) ? 0 : x;
}

double angle(Point3d &a, Point3d &b) {
	return acos((a.x * b.x + a.y * b.y + a.z * b.z) / (len(a) * len(b))) * 180
			/ 3.1415;
}

double evalXY(GiNaC::ex ex, GiNaC::symbol x, GiNaC::symbol y, double X,
		double Y) {
	return GiNaC::ex_to<GiNaC::numeric>(
			GiNaC::subs(ex, GiNaC::lst { x == X, y == Y })).to_double();
}

double sgn(double x) {
	return x >= 0 ? 1 : -1;
}

void getPrimeFactors(int n, std::map<int, int> &factors) {

	std::map<int, int>::iterator it;

	int incr = 1;
	for (int i = 2; i <= sqrt(n); i += incr) {
		while (n % i == 0) {
			it = factors.find(i);
			if (it == factors.end()) {
				factors[i] = 1;
			} else {
				it->second++;
			}
			n = n / i;
		}

		if (i == 3) {
			incr++;
		}
	}
}

void printPrimeFactors(std::map<int, int> &factors) {
	std::map<int, int>::iterator it;
	for (it = factors.begin(); it != factors.end(); it++) {
		std::cout << it->first << " :: " << it->second << std::endl;
	}
}

void gcdPrimeFactors(std::map<int, int> &f1, std::map<int, int> &f2,
		std::map<int, int> &gcd) {
	std::map<int, int>::iterator it1;
	std::map<int, int>::iterator it2;
	for (it1 = f1.begin(); it1 != f1.end(); it1++) {
		it2 = f2.find(it1->first);
		if (it2 != f2.end()) {
			gcd[it1->first] =
					(it1->second <= it2->second) ? it1->second : it2->second;
		}
	}
}

int mergePrimeFactors(std::map<int, int> &factors) {
	int merge = 1;
	std::map<int, int>::iterator it;
	for (it = factors.begin(); it != factors.end(); it++) {
		merge *= pow(it->first, it->second);
	}
	return merge;
}

int splitPrimeFactors(std::map<int, int> &factors, int aspecth, int aspectw,
		int limit) {

	int split = 1;
	std::map<int, int>::iterator it;
	for (it = factors.begin(); it != factors.end(); it++) {
		int base = it->first;
		int power = it->second;
		while (power-- > 0) {

			int splitpre = split * base;
			std::cout << "split0=" << split << " splitpre=" << splitpre
					<< " splitpre * aspecth * splitpre * aspectw="
					<< (splitpre * aspecth * splitpre * aspectw) << std::endl;
			if (splitpre * aspecth * splitpre * aspectw > limit) {
				return split;
			}
			split = splitpre;
		}
	}
	return split;
}

void print_matrix(const std::string s, const gsl_matrix *m) {
	print_matrix(s, m, false);
}

void print_matrix(const std::string s, const gsl_matrix *m, bool matlab) {

	std::cout << s << "=" << std::endl;
	int status, n = 0;

	for (size_t i = 0; i < m->size1; i++) {
		for (size_t j = 0; j < m->size2; j++) {
			printf("%g ", normalize0(gsl_matrix_get(m, i, j)));
		}
		printf(matlab ? ";" : "\n");
	}
}

void print_vec(const std::string title, const gsl_vector *v) {
	print_vec(title, v, false);
}

void print_vec(const std::string title, const gsl_vector *v, bool matlab) {
	std::cout << title << "=" << std::endl;
	for (size_t i = 0; i < v->size; i++) {
		printf("%g ", normalize0(gsl_vector_get(v, i)));
		if (matlab) {
			printf(";");
		}
	}
	printf("\n");
}

void print_vec_complex(const std::string title, const gsl_vector_complex *v) {
	std::cout << title << "=" << std::endl;
	for (size_t i = 0; i < v->size; i++) {
		gsl_complex z = gsl_vector_complex_get(v, i);
		printf("%g + %gi\n", GSL_REAL(z), GSL_IMAG(z));
	}
	printf("\n");
}

void help() {
	printf("\nThis sample demonstrates Canny edge detection\n"
			"Call:\n"
			"    /.sfs [image_name -- Default is cat.jpeg]\n\n");
}

void gsl_error_handler(const char *reason, const char *file, int line,
		int gsl_errno) {
	std::cout << "reason=" << reason << " gsl_errno=" << gsl_errno << " sel_i="
			<< sel_i << " sel_j=" << sel_j << std::endl;
}

string addIndex(std::string s, int idx) {
	return s + std::to_string(idx);
}
;

double gsl_matrix_get0(gsl_matrix *m, int row, int col) {
	return normalize0(gsl_matrix_get(m, row, col));
}

double getParmDouble(String parm, CommandLineParser &parser) {
	double val = parser.get<double>(parm);
	std::cout << parm << ":" << val << std::endl;
	return val;
}

int getParmInt(String parm, CommandLineParser &parser) {
	int val = parser.get<int>(parm);
	std::cout << parm << ":" << val << std::endl;
	return val;
}

string getParmString(String parm, CommandLineParser &parser) {
	string val = parser.get<string>(parm);
	std::cout << parm << ":" << val << std::endl;
	return val;
}

double thresh(double x, double t) {
	return abs(x) > t ? 0.0 : x;
}

void matrix_times_vector(gsl_matrix *m, gsl_vector *v) {
	gsl_vector *r = gsl_vector_alloc(v->size);
	for (int k = 0; k < v->size; k++) {
		double s = 0;
		for (int l = 0; l < m->size2; l++) {
			s += gsl_matrix_get(m, k, l) * gsl_vector_get(v, l);
		}
		gsl_vector_set(r, k, s);
	}
	gsl_vector_memcpy(v, r);
	gsl_vector_free(r);

}

gsl_matrix* matrix_times_matrix(gsl_matrix *m0, gsl_matrix *m1) {
	gsl_matrix *r = gsl_matrix_alloc(m0->size1, m1->size2);
	for (int k = 0; k < m0->size1; k++) {
		for (int l = 0; l < m1->size2; l++) {
			double sum = 0;
			for (int k_ = 0; k_ < m0->size2; k_++) {
				sum += gsl_matrix_get(m0, k, k_) * gsl_matrix_get(m1, k_, l);
			}
			gsl_matrix_set(r, k, l, sum);
		}
	}
	return r;
}

gsl_matrix* matrix_inverse(gsl_matrix *pi0) {
	gsl_matrix *pi0_cp = gsl_matrix_alloc(pi0->size1, pi0->size2);
	gsl_matrix_memcpy(pi0_cp, pi0);

	gsl_matrix *pi0_inv = gsl_matrix_alloc(pi0->size1, pi0->size2);
	gsl_permutation *p = gsl_permutation_alloc(pi0->size1);
	int s;
	gsl_linalg_LU_decomp(pi0, p, &s);
	gsl_linalg_LU_invert(pi0, p, pi0_inv);

	gsl_matrix *id = matrix_times_matrix(pi0_cp, pi0_inv);

//        if (isSelectedPixel()) {
//            print_matrix("id=", id);
//        }
	gsl_matrix_free(id);
	gsl_matrix_free(pi0_cp);
	return pi0_inv;
}
