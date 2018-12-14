#include <iostream>
#include <math.h>
#include <cmath>
#include <ctime>

int main(void)
{
	//Initialize the clock and set the number of iterations
	clock_t start = clock();
	int N = 10000000;
	float dx = 1.0 / N;
	
	//Compute the integral
	float integral;
	for (int i = 0; i < N; i++)
	{
		float x;
		x = (i + .5) * dx;
		integral += (1.0 / sqrt(1.0-(x*x))) * dx;
	}
	integral = integral * 2.0;
	
	//Finalize the clock
	clock_t end = clock();
	float el = float(end-start) / CLOCKS_PER_SEC;
	
	//Print the results
	std::cout << "Integral: " << integral << std::endl;
	std::cout << "Number of iterations: " << N << std::endl;
	std::cout << "Elapsed time: " << el << " seconds" << std::endl;

	return 0;
}
