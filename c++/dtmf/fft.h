#include <iostream>
#include <fstream>
#include <complex>
#include <vector>
#include <string>
#include <cmath>
#include <climits>
#include <winbase.h>

#define SAMPLE_RATE 8000
typedef struct
{
	unsigned int row_freq;
	unsigned int col_freq;
} freq_pair;

static char key_name[] = { '1', '2', '3', 'A', '4', '5', '6', 'B','7', '8',
	         			   '9', 'C', '*', '0', '#', 'D' };

static freq_pair key_freq[] = { {697, 1209}, {697, 1336}, {697, 1477}, {697, 1633},
						        {770, 1209}, {770, 1336}, {770, 1477}, {770, 1633},
								{852, 1209}, {852, 1336}, {852, 1477}, {852, 1633},
                                {941, 1209}, {941, 1336}, {941, 1477}, {941, 1633} };


class FFT
{
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

	FFT(std::string filename)
	{
		std::ifstream in(filename, std::ios::binary);
		if (!in.is_open())
			error("Cannot open data file!");
		else
		{
			unsigned bytes, dots;
			in.seekg(40);
			in.read(reinterpret_cast<char *> (&bytes), sizeof(unsigned int));
			dots = bytes / sizeof(short);
			N = 1 << static_cast<unsigned> (ceil(log2(1.0 * dots)));
			if (N == 0 || (N & (N - 1)))
				error("N is not a power of two!");
			x.resize(N);
			for (unsigned int i = 0; i < N; ++i)
			{
				// reverse order for DIT-FFT
				unsigned int t = getReverse(i);
				short data;
				in.read(reinterpret_cast<char *> (&data), sizeof(short));
				x[t] = (1.0 * data - SHRT_MIN) / (SHRT_MAX - SHRT_MIN) - 0.5;
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

	void output()
	{
		double max = 0.0;
		unsigned int max_index = 0;
		for (unsigned int i = 0; i < 16; ++i)
		{
			unsigned int f1 = key_freq[i].row_freq * N / SAMPLE_RATE;
			unsigned int f2 = key_freq[i].col_freq * N / SAMPLE_RATE;
		    double amp = abs(x[f1]) + abs(x[f2]);
		    if (amp > max)
		    {
		    	max = amp;
		    	max_index = i;
		    }
		}
		std::cout << "The actual number of this file is " << key_name[max_index] << std::endl;
	}

	unsigned int getN()
	{
		return N;
	}
};





