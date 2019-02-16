#ifndef KEY_H_
#define KEY_H_

#define SAMPLE_RATE 8000
typedef struct
{
	unsigned int row_freq;
	unsigned int col_freq;
} freq_pair;

static char key_name[] = { '1', '2', '3', 'A', '4', '5', '6', 'B','7', '8',
	         			   '9', 'C', '*', '0', '#', 'D' };

static freq_pair key_freq[] = {{697, 1209}, {697, 1336}, {697, 1477}, {697, 1633},
                                {770, 1209}, {770, 1336}, {770, 1477}, {770, 1633},
                                {852, 1209}, {852, 1336}, {852, 1477}, {852, 1633},
                                {941, 1209}, {941, 1336}, {941, 1477}, {941, 1633}};

#endif
