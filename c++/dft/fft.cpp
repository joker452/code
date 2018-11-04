#include <iostream>
#include <fstream>
#include <complex>
#include <vector>
#include <string>
#include <cmath>
#include <winbase.h>

class FFT
{
	bool is_dit;
	unsigned int N;
	std::vector<std::complex<double>> x;

	// dit input and dif output order
	unsigned int getReverse(unsigned int order)
	{
		unsigned int reverse = 0, logN = 0;

		while (N >> ++logN);
		--logN;

		for (unsigned int i = 0; i < logN; ++i)
		{
			unsigned int left = 32 - logN + i;
			reverse += ((order << left) >> 31) << i;
		}

		return reverse;
	}

	// butterfly group
	void addCross(unsigned int dots)
	{
		unsigned int half_dots = dots >> 1;
		std::complex<double> cache;
		for (unsigned int i = 0; i < N; i += dots)
			for (unsigned int j = i; j < i + half_dots; ++j)
			{
				cache = x[j] + x[j + half_dots];
				x[j + half_dots] = x[j] - x[j + half_dots];
				x[j] = cache;
			}
	}

	// multiply rotate factor
	void multRotate(unsigned int dots)
	{
		unsigned int half_dots = dots >> 1;
		std::complex<double> w(1.0, 0.0);
		std::complex<double> w_dots(cos(2 * M_PI / dots), sin(-2 * M_PI / dots));

		for (unsigned int i = 0; i < half_dots; ++i)
		{
			for (unsigned int j = i + half_dots; j < N; j += dots)
				x[j] *= w;
			w *= w_dots;
		}
	}

	// error handler
	void error(std::string message)
	{
		std::cerr << message << std::endl;
		exit(1);
	}
public:

	FFT(std::string filename, bool is_dit)
	{
		std::ifstream in(filename, std::ios::binary);
		if (!in.is_open())
			error("Cannot open data file!");
		else
		{
			in.read(reinterpret_cast<char *> (&N), sizeof(unsigned int));
			if (N == 0 || (N & (N - 1)))
				error("N is not a power of two!");
			this->is_dit = is_dit;
			x.resize(N);
			for (unsigned int i = 0; i < N; ++i)
			{
				double real, imagine;
				// reverse order for DIT-FFT
				unsigned int t = (is_dit) ? getReverse(i): i;
				in.read(reinterpret_cast<char *> (&real), sizeof(double));
				in.read(reinterpret_cast<char *> (&imagine), sizeof(double));
				x[t] = std::complex<double>(real, imagine);
			}
		}
		in.close();
	}

	~FFT()
	{
		x.clear();
	}

	// DIT-FFT
	void dit()
	{
		for (unsigned int i = 2; i <= N; i <<= 1)
		{
			multRotate(i);
			addCross(i);
		}
	}

	// DIF-FFT
	void dif()
	{
		for (unsigned int i = N; i >= 2; i >>= 1)
		{
			addCross(i);
			multRotate(i);
		}
	}

	// calculate DFT directly
	void dft()
	{
		if (is_dit)
			error("Data stored in wrong order! Try it with is_dit=false.");
		std::vector<std::complex<double>> result;
		result.resize(N);

		for (unsigned int k = 0; k < N; ++k)
		{
			result[k] = 0;
			// wN0 and wNk
			std::complex<double> w(1.0, 0.0);
			std::complex<double> w_k(cos(2 * M_PI * k / N), sin(2 * M_PI * k/ N));

			for (unsigned int n = 0; n < N; ++n)
			{
				result[k] += x[n] * w;
				w *= w_k;
			}
		}
	}

	void output(std::string name)
	{
		std::ofstream out("name", std::ios::binary);

		if (!out.is_open())
			error("Cannot output result");
		for (unsigned int i = 0; i < N; ++i)
		{
			// reverse order for DIF-FFT
			unsigned int t = (is_dit)? i: getReverse(i);
			double real = x[t].real(), imagine = x[t].imag();
			out.write(reinterpret_cast<char *>(&real), sizeof(double));
			out.write(reinterpret_cast<char*>(&imagine), sizeof(double));
		}
		out.close();
	}

	unsigned int getN()
	{
		return N;
	}
};


int main()
{
	FFT f("data2.dat", false);
	LARGE_INTEGER begin, end, microseconds, freq;
	QueryPerformanceFrequency(&freq);

	QueryPerformanceCounter(&begin);
	f.dif();
	QueryPerformanceCounter(&end);
	microseconds.QuadPart = (end.QuadPart - begin.QuadPart) * 1000000 / freq.QuadPart;
	std::cout << f.getN() << " points DIF-FFT done in "
			  << microseconds.QuadPart << " microseconds" << std::endl;
	f.output("dif-out.dat");
	QueryPerformanceCounter(&begin);
	f.dft();
	QueryPerformanceCounter(&end);
	microseconds.QuadPart = (end.QuadPart - begin.QuadPart) * 1000000 / freq.QuadPart;
	std::cout << f.getN() << " points DFT done in "
			  << microseconds.QuadPart << " microseconds" << std::endl;
	// note if you want to do DIT-FFT, you need another instance of class FFT,
	// because the order in which the data is stored cannot be changed once decided.
	// also note that you always need an instance with is_dit = false to do DFT directly,
	// because DFT assumed the data isn't stored in reverse order.
	// uncomment the following lines to see the performance of DIT-FFT

	/*FFT g("data.dat", true);
	QueryPerformanceCounter(&begin);
	g.dit();
	QueryPerformanceCounter(&end);
	microseconds.QuadPart = (end.QuadPart - begin.QuadPart) * 1000000 / freq.QuadPart;
	std::cout << g.getN() << " points DIT-FFT done in "
	          << microseconds.QuadPart << " microseconds" << std::endl;
	g.output("dit-out.dat");
	g.dft(); // error! data stored in wrong order
	*/

	return 0;

}



