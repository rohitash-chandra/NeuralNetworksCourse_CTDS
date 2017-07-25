/*
 * myGaPcx.cpp
 *
 *  Created on: 11/08/2009
 *      Author: rohit
 */
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <ctime>


#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<ctype.h>

time_t TicTime;
time_t TocTime;
using namespace::std;

typedef vector<double> Layer;
typedef vector<double> Nodes;
typedef vector<double> Frame;
typedef vector<int> Sizes;
typedef vector<vector<double> > Weight;
typedef vector<vector<double> > Data;



 const int LayersNumber = 3; //total number of layers.

 const int MaxVirtLayerSize = 50;


const double MaxErrorTollerance = 0.20;
const double MomentumRate = 0;
const double Beta = 0;
int row ;
int col;
int layer;
int r;
int x;
int y;


 //#include "objective.h"    //objective function
  //#include "random.h"       //random number generator
 #include "RandomNumber.h"       //random number generator


//#define rosen          // choose the function:



#define EPSILON 1e-80

#define MAXFUN 50000000  //upper bound for number of function evaluations

#define MINIMIZE 1      //set 1 to minimize and -1 to maximize
#define LIMIT 1e-80     //accuracy of best solution fitness desired
#define KIDS 2          //pool size of kids to be formed (use 2,3 or 4)
#define M 1             //M+2 is the number of parents participating in xover (use 1)
#define family 2        //number of parents to be replaced by good individuals(use 1 or 2)
#define sigma_zeta 0.1
#define sigma_eta 0.1   //variances used in PCX (best if fixed at these values)

#define  PopSize 400

#define NPSize KIDS + 2   //new pop size

#define RandParent M+2     //number of parents participating in PCX


 #define MAXRUN 50       //number of runs each with different random initial population

double d_not[PopSize];

double seed,basic_seed;



int RUN;




//*************************************************


//-------------------------------------------------------

class Individual{

      friend class GeneticAlgorithmn;
      protected:

        Layer Chrome;
        double Fitness;
        Layer BitChrome;

       public:
        Individual()
        {

        }
        void print();


      };
//***************************************************
//typedef vector <Individual> Pop ;
//class GeneticAlgorithmn:RandomNumber
  class GeneticAlgorithmn :RandomNumber{
	friend class Individual;
    friend class TrainingExamples;
        friend class NeuralNetwork;
        friend class  Layers;
       protected:


   	 Individual Population[PopSize];
     int TempIndex[PopSize];

   	 Individual  NewPop[NPSize];
     int mom[PopSize];
     int list[NPSize];

              int MaxGen;


       int NumVariable;

       double BestFit;
       int BestIndex;
       int NumEval;

       int  kids;

       int TotalEval;

              double Error;

   double Cycles;
           bool Sucess;

       public:
       GeneticAlgorithmn(int stringSize)
        {
    	   NumVariable = stringSize;
    	   NumEval=0;
        }
       int GetEval(){
                return TotalEval;
                      }
              double GetCycle(){
                         return Cycles;
                               }
              double GetError(){
                        return Error;
                              }

              bool GetSucess(){
                                  return Sucess;
                                        }

       double Fitness() {return BestFit;}

       double  RandomWeights();


       double  RandomAddition();

       void PrintPopulation();


       int GenerateNewPCX(int pass);

       double Objective(Layer x);

       void  InitilisePopulation();

       void Evaluate();


       double  modu(double index[]);



       // calculates the inner product of two vectors

       double  innerprod(double Ind1[],double Ind2[]);

       double RandomParents();

       double MainAlgorithm(double RUN, ofstream &out1, ofstream &out2, ofstream &out3);

       double Noise();

       void  my_family();   //here a random family (1 or 2) of parents is created who would be replaced by good individuals

       void  find_parents() ;
       void  rep_parents() ;  //here the best (1 or 2) individuals replace the family of parents
       void sort();

   };

   //-------------------------------


double GeneticAlgorithmn::RandomWeights()
{
      int chance;
      double randomWeight;
      double NegativeWeight;
   // srand ( time(NULL) );

  
      chance =rand()%2;

      if(chance ==0){
      randomWeight =rand()%  100000;
        
      return randomWeight*0.01;
       }

      if(chance ==1){
      NegativeWeight =rand()% 100000;
   
      return NegativeWeight*0.01;
     }

 

}

double GeneticAlgorithmn::RandomAddition()
{
      int chance;
      double randomWeight;
      double NegativeWeight;
      chance =rand()%2;

      if(chance ==0){
      randomWeight =rand()% 100;
      return randomWeight*0.009;
       }

      if(chance ==1){
      NegativeWeight =rand()% 100;
      return NegativeWeight*-0.009;
     }

}



void GeneticAlgorithmn::InitilisePopulation()
    {

double x, y;

  for(int row = 0; row < PopSize  ; row++) {
   for(int col = 0; col < NumVariable ; col++){

	//   x=randomperc();    // x is a uniform random number in (0,1)
	   //	y=(-10.0)+(5.0*x); // the formula used is y=a+(b-a)*x if y should be a random number in (a,b)
      Population[row].Chrome.push_back(RandomWeights());}
  }
  for(int row = 0; row < NPSize  ; row++) {
     for(int col = 0; col < NumVariable ; col++)
        NewPop[row].Chrome.push_back(0);

  }
  }

void GeneticAlgorithmn::Evaluate()
{
	// solutions are evaluated and best id is computed

	  Population[0].Fitness= Objective( Population[0].Chrome);
       BestFit = Population[0].Fitness;
       BestIndex = 0;

        for(int row = 0; row < PopSize  ; row++)
        {
          Population[row].Fitness= Objective( Population[row].Chrome);
          if ((MINIMIZE * BestFit) > (MINIMIZE * Population[row].Fitness))
          	{
        	  BestFit = Population[row].Fitness;
        	  BestIndex = row;
          	}
        }
}


void GeneticAlgorithmn::PrintPopulation()
{
	for(int row = 0; row < PopSize  ; row++) {
	   for(int col = 0; col < NumVariable ; col++)
	      cout<< Population[row].Chrome[col]<<" ";
	      cout<<endl;
	}

	  for(int row = 0; row < PopSize  ; row++)
		  cout<< Population[row].Fitness<<endl;

	 cout<<" ---"<<endl;
	 cout<<BestFit<<"  "<<BestIndex<<endl;

		for(int row = 0; row < NPSize  ; row++) {
		   for(int col = 0; col < NumVariable ; col++)
		      cout<< NewPop[row].Chrome[col]<<" ";
		      cout<<endl;
		}

}


double GeneticAlgorithmn:: Objective(Layer x)
{
  int i,j,k;
  double fit, sumSCH;

  fit=0.0;

  for(j=0;j< x.size();j++)
    fit+=((j+1)*(x[j]*x[j]));

#ifdef ellip
  // Ellipsoidal function
  for(j=0;j<NumVariable;j++)
    fit+=((j+1)*(x[j]*x[j]));
#endif

#ifdef schwefel
  // Schwefel's function
  for(j=0; j<NumVariable; j++)
    {
      for(i=0,sumSCH=0.0; i<j; i++)
	sumSCH += x[i];
      fit += sumSCH * sumSCH;
    }
#endif

#ifdef rosen
  //  Rosenbrock's function
//   for(j=0; j<NumVariable-1; j++)
  //  fit += 100.0*(x[j]*x[j] - x[j+1])*(x[j]*x[j] - x[j+1]) + (x[j]-1.0)*(x[j]-1.0);

   //NumEval++;
#endif

 // #ifdef 6legkine

/* 
 double sum =0;






double x1=  x[0];
double y1 =  x[1];
double z1=  x[2];
double x2=  x[3];
double y2 =  x[4];
double z2 =  x[5];
double x3 =  x[6];
double y3 =  x[7];
double z3=  x[8];


fit = pow(sqrt(pow(z1 + 44701/250,2) + pow(y1 - 48689/125,2) + pow(x1 - 464141/1000,2)) -1250,2) +
pow(sqrt(pow(z2 + 178791/1000,2) + pow(y2 - 207131/1000,2) + pow(x2 - 569471/1000,2)) -1250,2) +
pow(sqrt(pow((z3 + 178741/1000),2) + pow((x3 - 210581/2000),2) + pow(y3 + 597151/1000,2)) - 1250,2)   +
pow(sqrt(pow((527634487835101058785.0/527929843904574123982.0*y1 - 1447341260250814120395.0/263964921952287061991.0*y2 + 2894977876571101305987.0/527929843904574123982.0*y3 - 524656443486500.0/263964921952287061991.0*x1*z2 + 524656443486500.0/263964921952287061991.0*x2*z1 + 524656443486500.0/263964921952287061991.0*x1*z3 - 524656443486500.0/263964921952287061991.0*x3*z1 - 524656443486500.0/263964921952287061991.0*x2*z3 + 524656443486500.0/263964921952287061991.0*x3*z2 + 2986/5),2)+
+ pow((527634487835101058785.0/527929843904574123982.0*z1 - 1447341260250814120395.0/263964921952287061991.0*z2 + 2894977876571101305987.0/527929843904574123982.0*z3 + 524656443486500.0/263964921952287061991.0*x1*y2 - 524656443486500.0/263964921952287061991.0*x2*y1 - 524656443486500.0/263964921952287061991.0*x1*y3 + 524656443486500.0/263964921952287061991.0*y1*x3 + 524656443486500.0/263964921952287061991.0*x2*y3 - 524656443486500.0/263964921952287061991.0*x3*y2 + 178601/1000),2)
+ pow((527634487835101058785.0/527929843904574123982.0*x1 - 1447341260250814120395.0/263964921952287061991.0*x2 + 2894977876571101305987.0/527929843904574123982.0*x3 + 524656443486500.0/263964921952287061991.0*y1*z2 - 524656443486500.0/263964921952287061991.0*y2*z1 - 524656443486500.0/263964921952287061991.0*y1*z3 + 524656443486500.0/263964921952287061991.0*z1*y3 + 524656443486500.0/263964921952287061991.0*y2*z3 - 524656443486500.0/263964921952287061991.0*y3*z2 + 210581/2000),2 )) - 1250 ,2) +
pow(sqrt(pow((1613341505600505108027.0/1319824609761435309955.0*z1 - 1506260658930409049544.0/263964921952287061991.0*z2 + 7237786398812975449648.0/1319824609761435309955.0*z3 + 83872477062700.0/263964921952287061991.0*x1*y2 - 83872477062700.0/263964921952287061991.0*x2*y1 - 83872477062700.0/263964921952287061991.0*x1*y3 + 83872477062700.0/263964921952287061991.0*y1*x3 + 83872477062700.0/263964921952287061991.0*x2*y3 - 83872477062700.0/263964921952287061991.0*x3*y2 + 8923/50),2)
+ pow((1613341505600505108027.0/1319824609761435309955.0*y1 - 1506260658930409049544.0/263964921952287061991.0*y2 + 7237786398812975449648.0/1319824609761435309955.0*y3 - 83872477062700.0/263964921952287061991.0*x1*z2 + 83872477062700.0/263964921952287061991.0*x2*z1 + 83872477062700.0/263964921952287061991.0*x1*z3 - 83872477062700.0/263964921952287061991.0*x3*z1 - 83872477062700.0/263964921952287061991.0*x2*z3 + 83872477062700.0/263964921952287061991.0*x3*z2 - 51743/250),2)
+ pow((1613341505600505108027.0/1319824609761435309955.0*x1 - 1506260658930409049544.0/263964921952287061991.0*x2 + 7237786398812975449648.0/1319824609761435309955.0*x3 + 83872477062700.0/263964921952287061991.0*y1*z2 - 83872477062700.0/263964921952287061991.0*y2*z1 - 83872477062700.0/263964921952287061991.0*y1*z3 + 83872477062700.0/263964921952287061991.0*z1*y3 + 83872477062700.0/263964921952287061991.0*y2*z3 - 83872477062700.0/263964921952287061991.0*y3*z2 + 71218/125),2)) - 1250 ,2)   +
pow(sqrt(pow((28064424610193676249.0/22953471474111918434.0*y1 - 14018788143245839819.0/11476735737055959217.0*y2 + 22926623150409921823.0/22953471474111918434.0*y3 - 10213636780500.0/11476735737055959217.0*x1*z2 + 10213636780500.0/11476735737055959217.0*x2*z1 + 10213636780500.0/11476735737055959217.0*x1*z3 - 10213636780500.0/11476735737055959217.0*x3*z1 - 10213636780500.0/11476735737055959217.0*x2*z3 + 10213636780500.0/11476735737055959217.0*x3*z2 - 48673/125),2)
+ pow((28064424610193676249.0/22953471474111918434.0*z1 - 14018788143245839819.0/11476735737055959217.0*z2 + 22926623150409921823.0/22953471474111918434.0*z3 + 10213636780500.0/11476735737055959217.0*x1*y2 - 10213636780500.0/11476735737055959217.0*x2*y1 - 10213636780500.0/11476735737055959217.0*x1*y3 + 10213636780500.0/11476735737055959217.0*y1*x3 + 10213636780500.0/11476735737055959217.0*x2*y3 - 10213636780500.0/11476735737055959217.0*x3*y2 + 178441/1000),2)
+ pow((28064424610193676249.0/22953471474111918434.0*x1 - 14018788143245839819.0/11476735737055959217.0*x2 + 22926623150409921823.0/22953471474111918434.0*x3 + 10213636780500.0/11476735737055959217.0*y1*z2 - 10213636780500.0/11476735737055959217.0*y2*z1 - 10213636780500.0/11476735737055959217.0*y1*z3 + 10213636780500.0/11476735737055959217.0*z1*y3 + 10213636780500.0/11476735737055959217.0*y2*z3 - 10213636780500.0/11476735737055959217.0*y3*z2 + 232227/500),2))- 1250 ,2)+
pow(sqrt(pow(x2-x1,2)+pow(y2-y1,2)+pow(z2-z1,2)) - sqrt(188120101193.0/500000.0),2) +
pow(sqrt(pow(x3-x1,2)+pow(y3-y1,2)+pow(z3-z1,2)) - sqrt(11968628213.0/25000.0),2) +
pow(sqrt(pow(x3-x2,2)+pow(y3-y2,2)+pow(z3-z2,2)) - sqrt(9349310123.0/500000.0),2); */

/*


double fitness;
double Pex =  Chromozone[label][0];
double Pey=  Chromozone[label][1];
double t =   (Chromozone[label][2] * (3.14159/180));
 

 fitness = pow((sqrt(pow(Pex,2)  + pow(Pey,2)) - 100),2)  
+  pow((sqrt(pow(Pex + (50 * cos(t)) - 200,2) + pow(Pey + (50  * sin(t)),2)) - 120),2) 
+ pow((sqrt(pow(Pex + (40 * cos(t)) - (40 * sin(t)),2) + pow(Pey + (40 * cos(t)) + (40 * sin(t)) - 200,2)) - 150),2);
    */

/*
double pi = 22/7;

  double t1=  x[0]*pi/180;
  double t2 =  x[1]*pi/180;
  double t3=  x[2]*pi/180;
  double t4=  x[3]*pi/180;
  double t5 =  x[4]*pi/180;
  double t6 =  x[5]*pi/180;


  fit = sqrt(pow( ( (cos(t1)* cos(t2)*cos(t3)* cos(t4)*cos(t5)* cos(t6)) *(0.004621816205 * cos(t1)* cos(t2)*cos(t3)* cos(t4)*cos(t5)* cos(t6) - 0.03189593457)
		  + 0.005975896176 * pow(sin(t1),2)* pow(sin(t2),2)*pow(sin(t3),2)*pow(sin(t4),2)*pow(sin(t5),2)*pow(sin(t6),2)),2)
		  +pow( (sin(t1)*sin(t2)*sin(t3)*sin(t4)*sin(t5)*sin(t6) *(0.004621816205*cos(t1)* cos(t2)*cos(t3)* cos(t4)*cos(t5)* cos(t6)-0.003189593457 )
				  +0.005975896176*cos(t1)* cos(t2)*cos(t3)* cos(t4)*cos(t5)* cos(t6) * sin(t1)*sin(t2)*sin(t3)*sin(t4)*sin(t5)*sin(t6)  ),2)
  + 0.005975896176*pow(sin(t1),2)* pow(sin(t2),2)*pow(sin(t3),2)*pow(sin(t4),2)*pow(sin(t5),2)*pow(sin(t6),2)
  + pow(0.7988 - 0.008930009088* cos(t1)* cos(t2)*cos(t3)* cos(t4)*cos(t5)* cos(t6),2)
   + pow((-0.0003 - 0.008930009088*sin(t1)*sin(t2)*sin(t3)*sin(t4)*sin(t5)*sin(t6)),2) +1.44168049);

*/
NumEval++;
 //#endif


  return(fit);


}
//------------------------------------------------------------------------

void GeneticAlgorithmn:: my_family()   //here a random family (1 or 2) of parents is created who would be replaced by good individuals
{
  int i,j,index;
  int swp;
  double u;

  for(i=0;i<PopSize;i++)
    mom[i]=i;

  for(i=0;i<family;i++)
    {
   //   u=randomperc();
  //    index=(u*(PopSize-i))+i;

	  index = (rand()%PopSize) +i;

	 // cout<<"is index  "<<index<<endl;
      if(index>(PopSize-1)) index=PopSize-1;
      swp=mom[index];
      mom[index]=mom[i];
      mom[i]=swp;
    }
}

void GeneticAlgorithmn::find_parents()   //here the parents to be replaced are added to the temporary sub-population to assess their goodness against the new solutions formed which will be the basis of whether they should be kept or not
{
  int i,j,k;
  double u,v;

 my_family();
//cout<<kids<<endl;
  for(j=0;j<family;j++)
    {
      for(i=0;i<NumVariable;i++)
 	NewPop[kids+j].Chrome[i] = Population[mom[j]].Chrome[i];

      NewPop[kids+j].Fitness = Objective(NewPop[kids+j].Chrome);

    }
}



void GeneticAlgorithmn::rep_parents()   //here the best (1 or 2) individuals replace the family of parents
{
  int i,j;
  for(j=0;j<family;j++)
    {
      for(i=0;i<NumVariable;i++)
       Population[mom[j]].Chrome[i]=NewPop[list[j]].Chrome[i];

      Population[mom[j]].Fitness = Objective(Population[mom[j]].Chrome);

    }
}


void GeneticAlgorithmn::sort()

{
  int i,j, temp;
  double dbest;

  for (i=0;i<(kids+family);i++) list[i] = i;

  if(MINIMIZE)
    for (i=0; i<(kids+family-1); i++)
      {
	dbest = NewPop[list[i]].Fitness;
	for (j=i+1; j<(kids+family); j++)
	  {
	    if(NewPop[list[j]].Fitness < dbest)
	      {
		dbest = NewPop[list[j]].Fitness;
		temp = list[j];
		list[j] = list[i];
		list[i] = temp;
	      }
	  }
      }
  else
    for (i=0; i<(kids+family-1); i++)
      {
	dbest = NewPop[list[i]].Fitness;
	for (j=i+1; j<(kids+family); j++)
	  {
	    if(NewPop[list[j]].Fitness > dbest)
	      {
		dbest = NewPop[list[j]].Fitness;
		temp = list[j];
		list[j] = list[i];
		list[i] = temp;
	      }
	  }
      }
}


//---------------------------------------------------------------------
double GeneticAlgorithmn::  modu(double index[])
{
  int i;
  double sum,modul;

  sum=0.0;
  for(i=0;i<NumVariable ;i++)
    sum+=(index[i]*index[i]);

  modul=sqrt(sum);
  return modul;
}

// calculates the inner product of two vectors
double GeneticAlgorithmn::  innerprod(double Ind1[],double Ind2[])
{
  int i;
  double sum;

  sum=0.0;

  for(i=0;i<NumVariable ;i++)
    sum+=(Ind1[i]*Ind2[i]);

  return sum;
}

int GeneticAlgorithmn::GenerateNewPCX(int pass)
{
  int i,j,num,k;
  double Centroid[NumVariable];
  double tempvar,tempsum,D_not,dist;
  double tempar1[NumVariable];
  double tempar2[NumVariable];
  double D[RandParent];
  double d[NumVariable];
  double diff[RandParent][NumVariable];
  double temp1,temp2,temp3;
  int temp;

  for(i=0;i<NumVariable;i++)
    Centroid[i]=0.0;

  // centroid is calculated here
  for(i=0;i<NumVariable;i++)
    {
      for(j=0;j<RandParent;j++)
	Centroid[i]+=Population[TempIndex[j]].Chrome[i];

      Centroid[i]/=RandParent;
    }
  // calculate the distace (d) from centroid to the index parent arr1[0]
  // also distance (diff) between index and other parents are computed
  for(j=1;j<RandParent;j++)
    {
      for(i=0;i<NumVariable;i++)
	{
	  if(j == 1)
	    d[i]=Centroid[i]-Population[TempIndex[0]].Chrome[i];
	  diff[j][i]=Population[TempIndex[j]].Chrome[i]-Population[TempIndex[0]].Chrome[i];
	}
      if (modu(diff[j]) < EPSILON)
	{
	  cout<< "RUN Points are very close to each other. Quitting this run   " <<endl;

	  return (0);
	}
    }
  dist=modu(d); // modu calculates the magnitude of the vector

  if (dist < EPSILON)
    {
	  cout<< "RUN Points are very close to each other. Quitting this run    " <<endl;

      return (0);
    }

  // orthogonal directions are computed (see the paper)
  for(i=1;i<RandParent;i++)
    {
      temp1=innerprod(diff[i],d);
      temp2=temp1/(modu(diff[i])*dist);
      temp3=1.0-pow(temp2,2.0);
      D[i]=modu(diff[i])*sqrt(temp3);
    }

  D_not=0;
  for(i=1;i<RandParent;i++)
    D_not+=D[i];

  D_not/=(RandParent-1); //this is the average of the perpendicular distances from all other parents (minus the index parent) to the index vector

  // Next few steps compute the child, by starting with a random vector
  for(j=0;j<NumVariable;j++)
    {
      tempar1[j]=noise(0.0,(D_not*sigma_eta));
      //tempar1[j] = Noise();
      tempar2[j]=tempar1[j];
    }

  for(j=0;j<NumVariable;j++)
    {
      tempar2[j] = tempar1[j]-((innerprod(tempar1,d)*d[j])/pow(dist,2.0));
    }

  for(j=0;j<NumVariable;j++)
    tempar1[j]=tempar2[j];

  for(k=0;k<NumVariable;k++)
    NewPop[pass].Chrome[k]=Population[TempIndex[0]].Chrome[k]+tempar1[k];

 tempvar=noise(0.0,(sigma_zeta));

 // tempvar =Noise();
  for(k=0;k<NumVariable;k++)
      NewPop[pass].Chrome[k] += (tempvar*d[k]);

  // the child is included in the newpop and is evaluated

/*int count = 1;
  for(k=0;k<NumVariable;k++){
      if(( NewPop[pass].Chrome[k]<0)&&( NewPop[pass].Chrome[k]>360));{
    	 //  NewPop[pass].Chrome[k] = rand()%  360;
         count++;

      }
  }*/

   //if(count==NumVariable){
	//   cout<<"yes"<<endl;
  // return (1);

  // }

 // NewPop[pass].Fitness = Objective(NewPop[pass].Chrome);
  return (1);
}




//------------------------------------------------------------------------
double GeneticAlgorithmn::  RandomParents()
{

	int i,j,index;
	  int swp;
	  double u;
	  int delta;

	  for(i=0;i<PopSize;i++)
	    TempIndex[i]=i;

	  swp=TempIndex[0];
	  TempIndex[0]=TempIndex[BestIndex];  // best is always included as a parent and is the index parent
	                       // this can be changed for solving a generic problem
	 TempIndex[BestIndex]=swp;

	  for(i=1;i<RandParent;i++)  // shuffle the other parents
	    {
	    //u=randomperc();
	    index=(rand()%PopSize)+i;

	    if(index>(PopSize-1)) index=PopSize-1;
	    swp=TempIndex[index];
	    TempIndex[index]=TempIndex[i];
	    TempIndex[i]=swp;
	    }


}


void TIC( void )
{ TicTime = time(NULL); }

void TOC( void )
{ TocTime = time(NULL); }

double StopwatchTimeInSeconds()
{ return difftime(TocTime,TicTime); }



double GeneticAlgorithmn:: MainAlgorithm(double RUN, ofstream &out1, ofstream &out2, ofstream &out3 )
{

//for(int g =0; g < 10; g++){


	clock_t start = clock();
	double tempfit =0;
    int count =0;
    int tag;
      kids = KIDS;

    int gen = MAXFUN/kids;

    basic_seed=0.4122;   //arbitrary choice
/*
    out2<<"             Initial Parameters \n\n";
    out2<<"Population size : \n"<<PopSize;
    out2<<"Number of variables : \n"<<NumVariable;
    out2<<"Pool size of kids formed by PCX : \n"<<KIDS;
    out2<<"Number of parents participating in PCX : \n"<<RandParent;
     out2<<"Number of parents to be replaced by good kids : \n"<<family;
     out2<<"Sigma eta :  \n"<<sigma_eta ;
     out2<<"Sigma zeta : \n"<<sigma_zeta ;
     out2<<"Best fitness required :  \n"<<LIMIT ;
     out2<<"Number of runs desired :  \n\n"<<MAXRUN ;
*/
    //for(RUN=1;RUN<=MAXRUN;RUN++)
   //   {

  //	  printf("run... ");
        seed=basic_seed+(1.0-basic_seed)*(double)(RUN-1)/(double)MAXRUN;
        if(seed>1.0) printf("\n warning!!! seed number exceeds 1.0");
          randomize(seed);

	InitilisePopulation();
    Evaluate();

    tempfit=Population[BestIndex].Fitness;

   for(count=1;((count<=gen)&&(tempfit>=LIMIT));count++)
   	{
    //   cout<<count<<endl;

	   RandomParents();           //random array of parents to do PCX is formed

	  // for(int i = 0; i < PopSize; i++)
         //  cout<<TempIndex[i]<< " ";
         //  cout<<endl;

	    for(int i=0;i<kids;i++)
	   	     {
	    	//tag=0;
	    	tag = GenerateNewPCX(i);
	    	/*  while(tag==1){
	    		 tag = GenerateNewPCX(i); //generate a child using PCX
cout<<tag<<endl;
	    	  }
*/
	   	    if (tag == 0) break;
	   	     }
	   	   if (tag == 0) break;

	        find_parents();  // form a pool from which a solution is to be
	   	                           //   replaced by the created child

	   		   sort();          // sort the kids+parents by fitness

	   		   rep_parents();   // a chosen parent is replaced by the child

	   		   //finding the best in the population
	   			  BestIndex=0;
	   			  tempfit=Population[0].Fitness;

	   			  //cout<<tempfit<<endl;
	   			  for(int i=1;i<PopSize;i++)
	   			    if((MINIMIZE * Population[i].Fitness) < (MINIMIZE * tempfit))
	   			      {
	   				tempfit=Population[i].Fitness;
	   				BestIndex=i;
	   			      }

	   			  // print out results after every 100 generations
	   			    if (((count%100)==0) || (tempfit <= LIMIT))
	   			   out1<<count*kids<<"    "<<tempfit<<endl;

    	}

   clock_t finish = clock();

   Cycles = ((double)(finish - start))/CLOCKS_PER_SEC;
cout<<Cycles<<" ----"<<endl;
   Error = tempfit;

   TotalEval= NumEval;
      //  cout<<" Run Number: "<<RUN<<endl;
  // out2<<"Best solution obtained after X function evaluations:"<<count*kids<<" "<<NumEval<<endl;
out2<<RUN<<"   ";
        for(int i=0;i<NumVariable;i++)
  	 out2<<Population[BestIndex].Chrome[i]<<"      ";

        out2<<"---->  "<< tempfit<<"     "<<count*kids<<"      "<<Cycles<<endl;

        //out2<<"Fitness of this best solution:"<<tempfit<<endl;

        cout<<"Best solution obtained after X function evaluations:"<<count*kids<<" "<<NumEval<<endl;

              for(int i=0;i<NumVariable;i++)
            	  cout<<Population[BestIndex].Chrome[i]<<" ";
              cout<<endl;

              cout<<"Fitness of this best solution:"<<tempfit<<" "<<StopwatchTimeInSeconds()<<endl;


 	//   PrintPopulation();

     // }//run
  // }
}


int main(void)
{

	int VSize =15;

	ofstream out1;
		out1.open("out1.txt");
	ofstream out2;
	     out2.open("out2.txt");
	 	ofstream out3;
	 	     out3.open("out3.txt");

	 	    Sizes EvalAverage;
	 	       	 Layer ErrorAverage;
	 	       	 Layer CycleAverage;

	 	       	 int MeanEval=0;
	 	       	 double MeanError=0;
	 	       double MeanCycle=0;

	 	       	 int EvalSum=0;
	 	       	 double ErrorSum=0;
	 	       	double CycleSum=0;

	 	       	int maxrun=5;
	for(RUN=1;RUN<=maxrun;RUN++)
	     {

//srand (0);
		GeneticAlgorithmn GenAlg(VSize);
	        GenAlg.MainAlgorithm(RUN,out1,out2, out3);


	        EvalAverage.push_back(GenAlg.GetEval());
	                 MeanEval+=GenAlg.GetEval();

	                 ErrorAverage.push_back(GenAlg.GetError());
	           	  MeanError+= GenAlg.GetError();
cout<<GenAlg.GetCycle()<<" ----------------"<<endl;
	           	  CycleAverage.push_back(GenAlg.GetCycle());
	           	      	  MeanCycle+= GenAlg.GetCycle();


	     }


	 MeanEval=MeanEval/EvalAverage.size();
	 MeanError=MeanError/ErrorAverage.size();
	 MeanCycle=MeanCycle/CycleAverage.size();

	  for(int a=0; a < EvalAverage.size();a++)
	EvalSum +=	(EvalAverage[a]-MeanEval)*(EvalAverage[a]-MeanEval);

	  EvalSum=EvalSum/ EvalAverage.size();
	  EvalSum = sqrt(EvalSum);

	  for(int a=0; a < CycleAverage.size();a++)
		  CycleSum +=	(CycleAverage[a]-MeanCycle)*(CycleAverage[a]-MeanCycle);

	  CycleSum=CycleSum/ CycleAverage.size();
	  CycleSum = sqrt(CycleSum);
	  CycleSum = 1.96*(CycleSum/sqrt(maxrun));
	  for(int a=0; a < ErrorAverage.size();a++)
   ErrorSum +=	(ErrorAverage[a]-MeanError)*(ErrorAverage[a]-MeanError);

	 ErrorSum=ErrorSum/ ErrorAverage.size();
	ErrorSum = sqrt(ErrorSum);
   ErrorSum = 1.96*(ErrorSum/sqrt(maxrun));
	out3<< "  & $"<<MeanEval<<"{\\pm "<<EvalSum<<"}$ & $"<<MeanError<<"{\\pm "<<ErrorSum<<"}$  &  $"<<MeanCycle<<"{\\pm "<<CycleSum<<"}$ & "<<ErrorAverage.size()<<"  \\\\"<<endl;
	EvalAverage.empty();
	ErrorAverage.empty();
	CycleAverage.empty();

	out1.close();
	out2.close();
	out3.close();


 return 0;

};


