/*
 * individual.cpp
 *
 *  Created on: 13/08/2009
 *      Author: rohit
 */

#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<ctype.h>

using namespace::std;

#include "RandomNumber.h"


//---------------------------------------------------
double RandomNumber::noise(double mu ,double sigma) /* normal noise with specified mean & std dev: mu & sigma */
{
    return((randomnormaldeviate()*sigma) + mu);
}

double RandomNumber::randomnormaldeviate() /* random normal deviate after ACM algorithm 267 / Box-Muller Method */
{
 //   double sqrt(), log(), sin(), cos();
  //  double randomperc();
    double t;

    if(rndcalcflag)
    {
        rndx1 = sqrt(- 2.0*log((double) randomperc()));
        t = 6.2831853072 * (double) randomperc();
        rndx2 = sin(t);
        rndcalcflag = 0;
        return(rndx1 * cos(t));
    }
    else
    {
        rndcalcflag = 1;
        return(rndx1 * rndx2);
    }
}

/* Fetch a single random number between 0.0 and 1.0 -  */
/* Subtractive Method . See Knuth, D. (1969), v. 2 for */
/* details.Name changed from random() to avoid library */
/* conflicts on some machines                          */
double RandomNumber:: randomperc()
{
    jrand++;
    if(jrand >= 55)
    {
        jrand = 1;
        advance_random();
    }
    return((double) oldrand[jrand]);
}


void RandomNumber::advance_random()
/* Create next batch of 55 random numbers */
{
    int j1;
    double new_random;

    for(j1 = 0; j1 < 24; j1++)
    {
        new_random = oldrand[j1] - oldrand[j1+31];
        if(new_random < 0.0) new_random = new_random + 1.0;
        oldrand[j1] = new_random;
    }
    for(j1 = 24; j1 < 55; j1++)
    {
        new_random = oldrand [j1] - oldrand [j1-24];
        if(new_random < 0.0) new_random = new_random + 1.0;
        oldrand[j1] = new_random;
    }
}

int RandomNumber:: flip(double prob) /* Flip a biased coin - true if heads */
{
    if(randomperc() <= prob)
        return(1);
    else
        return(0);
}

void RandomNumber:: initrandomnormaldeviate()
/* initialization routine for randomnormaldeviate */
{
    rndcalcflag = 1;
}

void RandomNumber:: randomize(double seed)  /* Get seed number for random and start it up */
{
    int j1;

    for(j1=0; j1<=54; j1++) oldrand[j1] = 0.0;
    jrand=0;
    warmup_random(seed);
    initrandomnormaldeviate();
}


void RandomNumber:: warmup_random(double random_seed) /* Get random off and running */
{
    int j1, ii;
    double new_random, prev_random;

    oldrand[54] = random_seed;
    new_random = 0.000000001;
    prev_random = random_seed;
    for(j1 = 1 ; j1 <= 54; j1++)
    {
        ii = (21*j1)%54;
        oldrand[ii] = new_random;
        new_random = prev_random-new_random;
        if(new_random<0.0) new_random = new_random + 1.0;
        prev_random = oldrand[ii];
    }

    advance_random();
    advance_random();
    advance_random();

    jrand = 0;
}
