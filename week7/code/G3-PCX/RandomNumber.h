
#ifndef  RandomNumber_H
#define RandomNumber_H


#include <vector>
typedef vector<double> Layer;


class RandomNumber{

	friend class GeneticAlgorithmn;

protected:

	  double oldrand[55];   /* Array of 55 random numbers */
	  int jrand;                 /* current random number */
	  double rndx1, rndx2;    /* used with random normal deviate */
      int rndcalcflag; /* used with random normal deviate */

public:
    void   initrandomnormaldeviate() ;

    void   warmup_random(double random_seed) ;

//---------------------------------------------------------
    double  noise(double mu ,double sigma);
    double randomnormaldeviate();
    double   randomperc();
    void  advance_random() ;
    int  flip(double prob) ;
    void   randomize(double seed);

};



#endif
