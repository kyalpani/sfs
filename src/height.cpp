#include "height.h"

Height::Height() {

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

    int normativeSize = mergePrimeFactors(gcd);
    std::cout << "merge gcd=" << normativeSize << std::endl;
    aspecth = rows / normativeSize;
    aspectw = cols / normativeSize;

    int split = splitPrimeFactors(gcd, aspecth, aspectw, blocksize);
    aspecth *= split;
    aspectw *= split;

    std::cout << "split=" << split << " aspecth=" << aspecth << " aspectw=" << aspectw << std::endl;

}

void Height::allocate_height_matrices(int sz, gsl_matrix ** A, gsl_matrix ** Acopy, gsl_matrix ** V, gsl_vector ** b, gsl_vector ** S, gsl_vector ** work, gsl_vector ** X) {
    *A     = gsl_matrix_alloc(sz, sz);
    *Acopy = gsl_matrix_alloc(sz, sz);
    *b     = gsl_vector_alloc(sz);
    *V     = gsl_matrix_alloc(sz, sz);
    *S     = gsl_vector_alloc(sz);
    *work  = gsl_vector_alloc(sz);
    *X     = gsl_vector_alloc(sz);

    gsl_matrix_set_all(*A    , 0);
    gsl_matrix_set_all(*Acopy, 0);
    gsl_vector_set_all(*b    , 0);
    gsl_matrix_set_all(*V, 0);
    gsl_vector_set_all(*S, 0);
    gsl_vector_set_all(*work, 0);
    gsl_vector_set_all(*X, 0);

}

void Height::print_height_matrices(gsl_matrix * A, gsl_matrix * Acopy, gsl_matrix * V, gsl_vector * b, gsl_vector * S, gsl_vector * work, gsl_vector * X) {
    print_matrix("A", Acopy);
    print_matrix("U", A);
    print_matrix("V", V);
    print_vec("S", S);
    print_vec("w", work);
    print_vec("b", b);
    print_vec("X", X);

}

void Height::free_height_matrices(gsl_matrix * A, gsl_matrix * Acopy, gsl_matrix * V, gsl_vector * b, gsl_vector * S, gsl_vector * work, gsl_vector * X) {
    gsl_matrix_free(A);
    gsl_matrix_free(Acopy);
    gsl_matrix_free(V);
    gsl_vector_free(b);
    gsl_vector_free(S);
    gsl_vector_free(work);
    gsl_vector_free(X);
}

void Height::balance_heights(int h, int w, std::function<double(int, int, int, int)> getRHS, std::function<void(gsl_vector *)> transferHeights) {

    int sz   = h * w - 1;
    int limi = h - 1;
    int limj = w - 1;

    gsl_matrix * A;
    gsl_matrix * Acopy;
    gsl_matrix * V;
    gsl_vector * b;
    gsl_vector * S;
    gsl_vector * work;
    gsl_vector * X;

    allocate_height_matrices(sz, &A, &Acopy, &V, &b, &S, &work, &X);

    std::cout << "limi=" << limi << " limj=" << limj << std::endl;
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

            std::cout << "i=" << i << " j=" << j << " si=" << si << " sj=" << sj << std::endl;
            gsl_vector_set(b, k, getRHS(i, j, si, sj));
        }
    }
    gsl_matrix_memcpy(Acopy, A);

    gsl_linalg_SV_decomp (A, V, S, work);
    gsl_linalg_SV_solve (A, V, S, b, X);

    transferHeights(X);

    print_height_matrices(A, Acopy, V, b, S, work, X);
    free_height_matrices (A, Acopy, V, b, S, work, X);
}

void Height::convert_normals_to_height_h0(int k, int m, int h, int w) {


    int topi = k * h;
    int topj = m * w;

    auto getRHS = [this, topi, topj](int i, int j, int si, int sj) {

        int reali = topi + i;
        int realj = topj + j;

        markSelectedPixel(reali, realj);
        double p = (double) this->getterZMatrix(9 , zMatrix, reali, realj);
        double q = (double) this->getterZMatrix(10, zMatrix, reali, realj);

        double rhs = sj * p * dx - si * q * dy;

        return rhs;
    };

    auto transferHeights = [this, topi, topj, h, w](gsl_vector * height) {

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {

                if (i == h - 1 && j == w - 1) {
                    continue;
                }

                int reali = topi + i;
                int realj = topj + j;
                hMatrix[reali][realj] = gsl_vector_get(height, i * h + j);
            }
        }

        // we need to fill in the last point in the matrix
        int lasti = topi + h - 1;
        int lastj = topj + w - 1;
        if (hMatrix[lasti][lastj - 1]) {
            double plast = (double) this->getterZMatrix(9 , zMatrix, lasti, lastj - 1);
            hMatrix[lasti][lastj] = plast * dx + hMatrix[lasti][lastj - 1].value();
        }
    };

    balance_heights(h, w, getRHS, transferHeights);
}

void Height::convert_normals_to_height_h1(int k, int m, int h, int w, int blockh, int blockw) {


    // h = the number of cell rows in level 1
    // w = the number of cell columns in level1
    // blockh = the height of each cell (number of subcells)
    // blockw = the width of each cell (number of subcells)

    // the actual horizontal number of pixels in each cell
    int realh = h * blockh;
    // the actual vertical number of pixels in each cell
    int realw = w * blockw;

    int topi = k * realh;
    int topj = m * realw;

    auto getRHS = [this, topi, topj, h, w, realh, realw](int i, int j, int si, int sj) {

        double pp = 0;
        int fixedj = topj + realw;
        if (sj > 0) {
            for (int ii = 0; ii < h; ii++) {
                int reali = topi + ii;
                std::cout << "reali=" << reali << " fixedj=" << fixedj << std::endl;
                markSelectedPixel(reali, fixedj);
                double p     = (double) this->getterZMatrix(9, zMatrix , reali, fixedj - 1);
                double z     = (double) this->getterHMatrix(0, hMatrix, reali, fixedj - 1);
                double znext = (double) this->getterHMatrix(0, hMatrix, reali, fixedj);
                double delta = z + p * dx - znext;
                pp += delta;
            }
        }
        pp /= h;

        double qq = 0;
        if (si > 0) {
            int fixedi = topi + realh;
            for (int jj = 0; jj < w; jj++) {
                int realj = topj + jj;
                markSelectedPixel(fixedi, realj);
                double q     = (double) this->getterZMatrix(10, zMatrix , fixedi - 1, realj);
                double z     = (double) this->getterHMatrix(0 , hMatrix, fixedi - 1, realj);
                double znext = (double) this->getterHMatrix(0 , hMatrix, fixedi, realj);
                double delta = z - q * dy - znext;
                qq += delta;
            }
        }

        qq /= w;

        double rhs = si * qq + sj * pp;
        return rhs;
    };

    auto transferHeights = [this, topi, topj, h, w, blockh, blockw](gsl_vector * height) {

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {

                double h1 = gsl_vector_get(height, i * h + j);
                // now add h1 to the heights of the previous iteration

                for (int ii = 0; ii < blockh; ii++) {
                    int reali = topi + i * blockh + ii;
                    for (int jj = 0; jj < blockw; jj++) {
                        int realj = topj + j * blockw + jj;
                        std::cout << "reali=" << reali << " realj=" << realj << std::endl;
                        if (hMatrix[reali][realj]) {
                            hMatrix[reali][realj].value() += h1;
                        }
                    }
                }
            }
        }
    };

    balance_heights(h, w, getRHS, transferHeights);
}

void Height::convert_normals_to_height_h0() {
    for (int k = 0; k < 50; k++) {
        for (int m = 0; m < 50; m++) {
            convert_normals_to_height_h0(k, m, aspecth, aspectw);
        }
    }
}

void Height::convert_normals_to_height_h1() {
    for (int k = 0; k < 9; k++) {
        for (int m = 0; m < 9; m++) {
            convert_normals_to_height_h1(k, m, 5, 5, aspecth, aspectw);
        }
    }
}

void Height::convert_normals_to_height_h2() {

    int h = 10;
    int w = 10;

    int blockh = 5 * aspecth;
    int blockw = 5 * aspectw;

    auto getRHS = [this, blockh, blockw](int i, int j, int si, int sj) {

        int topi = i * blockh;
        int topj = j * blockw;

        double pp = 0;
        int fixedj = topj + blockw;
        if (sj > 0) {
            for (int ii = 0; ii < blockh; ii++) {
                int reali = topi + ii;
                std::cout << "reali=" << reali << " fixedj=" << fixedj << std::endl;
                markSelectedPixel(reali, fixedj);
                double p     = (double) this->getterZMatrix(9, zMatrix , reali, fixedj - 1);
                double z     = (double) this->getterHMatrix(0, hMatrix, reali, fixedj - 1);
                double znext = (double) this->getterHMatrix(0, hMatrix, reali, fixedj);
                double delta = z + p * dx - znext;
                pp += delta;
            }
        }

        pp /= blockh;

        double qq = 0;
        if (si > 0) {
            int fixedi = topi + blockh;
            for (int jj = 0; jj < blockw; jj++) {
                int realj = topj + jj;
                markSelectedPixel(fixedi, realj);
                double q     = (double) this->getterZMatrix(10, zMatrix , fixedi - 1, realj);
                double z     = (double) this->getterHMatrix(0 , hMatrix, fixedi - 1, realj);
                double znext = (double) this->getterHMatrix(0 , hMatrix, fixedi, realj);
                double delta = z - q * dy - znext;
                qq += delta;
            }
        }

        qq /= blockh;

        double rhs = si * qq + sj * pp;
        return rhs;
    };

    auto transferHeights = [this, h, w, blockh, blockw](gsl_vector * height) {

        for (int i = 0; i < h; i++) {
            int topi = i * blockh;
            for (int j = 0; j < w; j++) {

                int topj = j * blockw;
                double h2 = gsl_vector_get(height, i * h + j);

                // now add h2 to the heights of the previous iterations

                for (int ii = 0; ii < blockh; ii++) {
                    int reali = topi + ii;
                    for (int jj = 0; jj < blockw; jj++) {
                        int realj = topj + jj;
                        std::cout << "reali=" << reali << " realj=" << realj << std::endl;
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

void Height::convert_normals_to_height() {
  hMatrix.resize(boost::extents[rows][cols]);
  convert_normals_to_height_h0();
  convert_normals_to_height_h1();
  convert_normals_to_height_h2();
}
