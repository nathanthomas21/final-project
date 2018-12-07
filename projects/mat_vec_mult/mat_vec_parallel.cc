#include <cstdio>
#include <omp.h>

void matvec(double *a, double *x, double *y, int n);
void zero_array(double *v, int n);

int main(int argc, char* argv[])
{
	double et = omp_get_wtime();

  // size of arrays
  int n = 1000; 

  // allocate memory
  double *a = new double[n*n];
  double *x = new double[n];
  double *y = new double[n];


#pragma omp parallel
{
  // initialize input arrays
#pragma omp for
  for (int i = 0; i < n*n; ++i)
    a[i] = 0.0;
#pragma omp for
  for (int i = 0; i < n; ++i)
  {
    a[i*n+i] = 4.0;
    if (i > 0) a[i*n+i-1] = -1.0;
    if (i < n-1) a[i*n+i+1] = -1.0;
    x[i] = 1.0;
    y[i] = 0.0;
  }

  // zero out output array
  zero_array(y, n);

  // compute the matrix vector product
  matvec(a, x, y, n);
}

	et = omp_get_wtime() - et;
  // print out the first two elements of y
  printf("y[0]=%8.4e and y[1]=%8.4e\n", y[0], y[1]);
	printf("Elapsed time: %8.8f seconds\n", et);
	printf("Number of elements in array %8.0f\n", float(n));
  double b = y[1];
  // free memory
  delete [] a;
  delete [] x;
  delete [] y;

  return 0;
}

void zero_array(double *v, int n)
{
  for (int i = 0; i < n; ++i)
    v[i] = 0.0;
}

void matvec(double *a, double *x, double *y, int n)
{
#pragma omp for
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < n; ++j)
    {
      y[i] += a[i*n+j]*x[j];
    }
  }
}

