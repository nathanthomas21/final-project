#include <iostream>
#include <math.h>
#include <cmath>
#include <ctime>

/*
int f(float n)
{
	float product = 1 / sqrt(1-(n*n));
	std::cout << "Product: " << product << std::endl;
	return product;
}
*/

int main(void)
{
	clock_t start = clock();
	int N = 10000000;
	float dx = 1.0 / N;
	
/*
	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;
	int chunk = N / size;
*/
	float integral;
	for (int i = 0; i < N; i++)
	{
		float x;
		x = (i + .5) * dx;
		//std::cout << "X value: " << x << std::endl;
		integral += (1.0 / sqrt(1.0-(x*x))) * dx;
	}
	integral = integral * 2.0;
	clock_t end = clock();
	float el = float(end-start) / CLOCKS_PER_SEC;
	std::cout << "Integral: " << integral << std::endl;
	std::cout << "Number of iterations: " << N << std::endl;
	std::cout << "Elapsed time: " << el << " seconds" << std::endl;

	return 0;
}
