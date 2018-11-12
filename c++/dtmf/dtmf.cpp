#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <climits>
#include <cmath>
#include "fft.h"
typedef struct
{

} KEY;


class DTMF
{
	std::vector<double> x;
	unsigned N;

	void error(std::string message)
	{
		std::cerr << message << std::endl;
	}

	double Goertzel(unsigned int start, unsigned int k, unsigned int N)
	{
		double w = cos(2 * M_PI * k / N);
		double v[2];
		v[0] = x[start + 1] + 2.0 * w * x[start + 0];
		v[1] = x[start + 2] + 2.0 * w * v[0];
		unsigned int i = 2;

		for (; i < N; ++i)
			v[i % 2] = x[start + i + 1] + 2.0 * w * v[(i - 1) % 2] - v[(i - 2) % 2];
		return sqrt((v[i % 2] - v[(i - 1) % 2] * w) *
				    (v[i % 2] - v[(i - 1) % 2] * w) +
					(v[(i - 1) % 2] * sin(2 * M_PI * k / N)) *
					(v[(i - 1) % 2] * sin(2 * M_PI * k / N)));
	}

public:

	DTMF(std::string filename, bool need_pad)
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
			N = (need_pad)? 1 << static_cast<unsigned> (ceil(log2(1.0 * dots))): dots;

		    x.resize(N);
			for (unsigned int i = 0; i < dots; ++i)
			{
				short data;
				in.read(reinterpret_cast<char *> (&data), sizeof(short));
				x[i] = (1.0 * data - SHRT_MIN) / (SHRT_MAX - SHRT_MIN) - 0.5;
			}
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
		    double amp = Goertzel(0, f1, N) + Goertzel(0, f2, N);
		    if (amp > max)
		    {
		    	max = amp;
		    	max_index = i;
		    }
		}
		std::cout << "The actual number of this file is " << key_name[max_index] << std::endl;
	}

	void detect()
	{
		const unsigned int length = 1 << 9;
		char *keys = new char[N / length]();
		for (unsigned int k = 0; k < N / length; ++k)
		{
			unsigned int max_index = 0;
			double max = 0.0;
			for (unsigned int i = 0; i < 16; ++i)
			{
				unsigned int f1 = key_freq[i].row_freq * length / SAMPLE_RATE;
				unsigned int f2 = key_freq[i].col_freq * length / SAMPLE_RATE;
				double amp = Goertzel(k * length, f1, length) + Goertzel(k * length, f2, length);
				if (amp > max)
				{
					max = amp;
					max_index = i;
				}
			}
			if (max > 10.0)
				keys[k] = key_name[max_index];
		}

		bool begin = false;
		for (unsigned i = 1; i < N / length; ++i)
		{
			if (begin == false && !keys[i - 1] && keys[i])
				begin = true;

			if (begin && keys[i] == keys[i + 1])
			{
				std::cout << keys[i] << " ";
			}
			begin = false;
		}
	}
};

int main(int argc, char **argv)
{
	for (int i = 1; i < argc; ++i)
	{
		std::string name = argv[i];
		DTMF f(argv[i], false);
		f.detect();
	}
}
