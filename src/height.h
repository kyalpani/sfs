#ifndef __HEIGHT_H__
#define __HEIGHT_H__

#include "sfs.h"
#include "point6d.h"

typedef boost::multi_array<boost::optional<double>, 2> HeightMatrix;

class Height {

public:

  SFS & sfs;
  HeightMatrix hMatrix;
  double dx;
  double dy;
  int rows;
  int cols;
  int blocksize;

  Height();
  void convert_normals_to_height();

private:

  int aspecth;
  int aspectw;

  Point6dMatrix zMatrix;
  void allocate_height_matrices(int sz, gsl_matrix ** A, gsl_matrix ** Acopy, gsl_matrix ** V, gsl_vector ** b, gsl_vector ** S, gsl_vector ** work, gsl_vector ** X);
  void print_height_matrices(gsl_matrix * A, gsl_matrix * Acopy, gsl_matrix * V, gsl_vector * b, gsl_vector * S, gsl_vector * work, gsl_vector * X);
  void free_height_matrices(gsl_matrix * A, gsl_matrix * Acopy, gsl_matrix * V, gsl_vector * b, gsl_vector * S, gsl_vector * work, gsl_vector * X);
  void balance_heights(int h, int w, std::function<double(int, int, int, int)> getRHS, std::function<void(gsl_vector *)> transferHeights);
  void convert_normals_to_height_h0(int k, int m, int h, int w);
  void convert_normals_to_height_h1(int k, int m, int h, int w, int blockh, int blockw);
  void convert_normals_to_height_h0();
  void convert_normals_to_height_h1();
  void convert_normals_to_height_h2();
};

#endif
