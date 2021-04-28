/*
 * main.cpp
 *
 *  Created on: Apr 25, 2021
 *      Author: kyal
 */




#include <stdio.h>
#include <gsl/gsl_errno.h>
#include "sfs.h"
#include "output.h"

int main(int argc, const char **argv) {

	gsl_set_error_handler(&gsl_error_handler);

	SFS sfs(argc, argv);
	sfs.recover_shape();

	Output output(sfs);
	output.output_shape();
}
