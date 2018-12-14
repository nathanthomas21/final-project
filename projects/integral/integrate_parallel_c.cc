#include <iostream>
#include <math.h>
#include <cmath>
#include <ctime>
#include <omp.h>


int main(void)
{
	//Initialize clock and set number of iterations
	clock_t start = clock();
	int N = 10000000;
	float dx = 1.0 / N;

	float integral = 0.0;

	//Calculate the integral in parallel
	#pragma omp parallel
	{
		float integral_p = 0.0;
		#pragma omp for
		for (int i = 0; i < N; i++)
		{
			float x;
			x = (i + .5) * dx;
			integral_p += (1.0 / sqrt(1.0-(x*x))) * dx;
		}
		integral += integral_p * 2.0;
	}
	
	//Finish the timer
	clock_t end = clock();
	float el = float(end-start) / CLOCKS_PER_SEC;
	
	//Print the results
	std::cout << "Integral: " << integral << std::endl;
	std::cout << "Number of iterations: " << N << std::endl;
	std::cout << "Elapsed time: " << el << " seconds" << std::endl;

	return 0;
}
