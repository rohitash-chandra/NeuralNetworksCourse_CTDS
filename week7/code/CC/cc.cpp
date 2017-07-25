/*

 Title: Cooperative Coevolution in C++ based on G3-PCX Evolutionary Algorithm
  Author: Rohitash Chandra, "c.rohitash (at) gmail.com"
  Website: www.softwarefoundationfiji.org/rohitash

  Compilor: G++ in Linux

Cooperative Coevolution: http://en.wikipedia.org/wiki/Cooperative_coevolution

G3-PCX Evolutionary Algorithm is used as the sub-populations of Cooperative Coveolution.

This program build on the G3-PCX C code from here: http://www.iitk.ac.in/kangal/codes.shtml

The paper on the G3-PCX: " Kalyanmoy Deb, Ashish Anand, and Dhiraj Joshi, A Computationally Efficient Evolutionary Algorithm for Real-Parameter Optimization, Evolutionary Computation 2002 10:4, 371-395"

 In this code, CC is used for solving general function optimisation problems (Sphere, Rosenbrock, Rastrigin).

 This code has been used to train feedforward and recurrent neural networks, publications here:
http://softwarefoundationfiji.org/rohitash/Scientifc_Publications

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



//-----------------------




#define EPSILON 1e-50


#define MINIMIZE 1      //set 1 to minimize and -1 to maximize
#define LIMIT 1e-20     //accuracy of best solution fitness desired
#define KIDS 2          //pool size of kids to be formed (use 2,3 or 4)
#define M 1             //M+2 is the number of parents participating in xover (use 1)
#define family 2        //number of parents to be replaced by good individuals(use 1 or 2)
#define sigma_zeta 0.1
#define sigma_eta 0.1   //variances used in PCX (best if fixed at these values)





#define NPSize KIDS + 2   //new pop size

#define RandParent M+2     //number of parents participating in PCX


double seed,basic_seed;


class Individual{

       public:

        Layer Chrome;
        double Fitness;
        Layer BitChrome;

       public:
        Individual()
        {

        }
        void print();


      };

//****************************************

typedef vector<double> Nodes;
  class GeneticAlgorithmn// :public virtual RandomNumber
  {
       public:

        int PopSize;

     	vector<Individual> Population;

   	 Layer d_not;

     	Sizes TempIndex ;

     	vector<Individual> NewPop;

     	Sizes mom ;
     	Sizes list;

              int MaxGen;


       int NumVariable;

       double BestFit;
       int BestIndex;
       int NumEval;

       int  kids;

       public:
        GeneticAlgorithmn(int stringSize )
        {
     	   NumVariable = stringSize;
          NumEval=0;
           BestIndex = 0;

         }
        GeneticAlgorithmn()
       {
             BestIndex = 0;
        }



       double Fitness() {return BestFit;}

       double  RandomWeights();


       double  RandomAddition();

       void PrintPopulation();


       int GenerateNewPCX(int pass,  double Mutation, int depth);

       double Objective(Layer x);

       void  InitilisePopulation(int popsize);

       void Evaluate();


       double  modu(double index[]);



       // calculates the inner product of two vectors

       double  innerprod(double Ind1[],double Ind2[]);

       double RandomParents();

       double MainAlgorithm(double RUN, ofstream &out1, ofstream &out2, ofstream &out3);

       double Noise();

double  rand_normal(double mean, double stddev);

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
      chance =rand()%2;

      if(chance ==0){
      randomWeight =rand()%  100000;
      return randomWeight*0.00005;
       }

      if(chance ==1){
      NegativeWeight =rand()% 100000;
      return NegativeWeight*-0.00005;
     }

}


double GeneticAlgorithmn::rand_normal(double mean, double stddev) { //Box Muller Random Numbers
    static double n2 = 0.0;
    static int n2_cached = 0;
    if (!n2_cached) {
       // choose a point x,y in the unit circle uniformly at random
     double x, y, r;
     do {
  //  scale two random integers to doubles between -1 and 1
   x = 2.0*rand()/RAND_MAX - 1;
   y = 2.0*rand()/RAND_MAX - 1;

    r = x*x + y*y;
   } while (r == 0.0 || r > 1.0);

        {
       // Apply Box-Muller transform on x, y
        double d = sqrt(-2.0*log(r)/r);
      double n1 = x*d;
      n2 = y*d;

       // scale and translate to get desired mean and standard deviation

      double result = n1*stddev + mean;

        n2_cached = 1;
        return result;
        }
    } else {
        n2_cached = 0;
        return n2*stddev + mean;
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



void GeneticAlgorithmn::InitilisePopulation(int popsize)
    {

double x, y;



Individual Indi ;

	   NumEval=0;
           BestIndex = 0;
           PopSize = popsize;



      for (int i = 0; i < PopSize; i++){
         TempIndex.push_back(0);
         mom.push_back(0);
          d_not.push_back(0); }

      for (int i = 0; i < NPSize; i++){
         list.push_back(0);}

 for(int row = 0; row < PopSize  ; row++)
        Population.push_back(Indi);

  for(int row = 0; row < PopSize  ; row++) {
   for(int col = 0; col < NumVariable ; col++){

      Population[row].Chrome.push_back(RandomWeights());}
  }


 for(int row = 0; row < NPSize  ; row++)
        NewPop.push_back(Indi);

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
	 for(int row = 0; row < PopSize/5   ; row++) {
	   for(int col = 0; col < NumVariable ; col++)
	      cout<< Population[row].Chrome[col]<<" ";
	      cout<<endl;
	}

	  for(int row = 0; row < PopSize/5  ; row++)
		  cout<< Population[row].Fitness<<endl;

	 cout<<" ---"<<endl;
	 cout<<BestFit<<"  "<<BestIndex<<endl;

		/*for(int row = 0; row < NPSize  ; row++) {
		   for(int col = 0; col < NumVariable ; col++)
		      cout<< NewPop[row].Chrome[col]<<" ";
		      cout<<endl;
		}*/

}


double GeneticAlgorithmn:: Objective(Layer x)
{

  return 0;
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

int GeneticAlgorithmn::GenerateNewPCX(int pass,  double Mutation, int depth)
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

  //cout<<Centroid[i]<<" --- "<<RandParent<<"  ";
          // if(isnan(Centroid[i])) return 0;
    }
  //cout<<endl;

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

 if (isnan(diff[j][i])  )
	{
	  cout<< "`diff nan   " <<endl;
             diff[j][i] = 1;
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
      if((modu(diff[i])*dist) == 0){
       cout<<" division by zero: part 1"<<endl;
         temp2=temp1/(1);
        }
      else{
      temp2=temp1/(modu(diff[i])*dist);}

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
    //  tempar1[j]=noise(0.0,(D_not*sigma_eta));
      tempar1[j] = rand_normal(0, D_not*sigma_eta);
      tempar2[j]=tempar1[j];
    }

  for(j=0;j<NumVariable;j++)
    {
      if ( pow(dist,2.0)==0){
     cout<<" division by zero: part 2"<<endl;
       tempar2[j] = tempar1[j]-((innerprod(tempar1,d)*d[j])/1);
   }
      else
      tempar2[j] = tempar1[j]-((innerprod(tempar1,d)*d[j])/pow(dist,2.0));
    }

  for(j=0;j<NumVariable;j++)
    tempar1[j]=tempar2[j];

  for(k=0;k<NumVariable;k++)
    NewPop[pass].Chrome[k]=Population[TempIndex[0]].Chrome[k]+tempar1[k];

 //tempvar=noise(0.0,(sigma_zeta));
   tempvar=    rand_normal(0, sigma_zeta);


 // tempvar =Noise();


  for(k=0;k<NumVariable;k++){
      NewPop[pass].Chrome[k] += (tempvar*d[k]);

}

  double random = rand()%10;

Layer Chrome(NumVariable);

for(k=0;k<NumVariable;k++){
if(!isnan(NewPop[pass].Chrome[k] )){
 Chrome[k]=NewPop[pass].Chrome[k] ;
}
else
  NewPop[pass].Chrome[k] =  RandomAddition();

   }




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



//--------------------------------------------------------------------------------------------


typedef vector<GeneticAlgorithmn> GAvector;

class Table{

	//  friend class  CoEvolution ;
    //  friend class CombinedEvolution;
	    public:

       //double   SingleSp[100];
       Layer   SingleSp ;



};

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

typedef vector<Table> TableVector;

class CoEvolution :      public   GeneticAlgorithmn   //Cooperative Coevolution class
{



public:

      int  NoSpecies;// No. Subpopulations
    GAvector Species ;
    TableVector  TableSp; //Table of Best Solutions from each Sub-Population
     int PopSize; //Sub-pop size
    vector<bool> NotConverged;
       Sizes SpeciesSize;
       Layer   Individual;
       Layer  BestIndividual;
      double bestglobalfit;
      Data TempTable;
int TotalEval;
     int TotalSize;
int ProbNum;
     int SingleGAsize;
     double Train;
          double Test;
     int kid;

    CoEvolution(){

    }

       void   MainProcedure(bool bp,   int RUN, ofstream &out1, ofstream &out2, ofstream &out3, double mutation,int depth );

     void  InitializeSpecies(int popsize);
     void  EvaluateSpecies();
     void    GetBestTable(int sp);
     void PrintSpecies();

     void   Join();
     double    ObjectiveFunc(Layer x);
     void Print();

     void   sort(int s );

     void  find_parents(int s);

     void EvalNewPop(int pass, int s );

     void  rep_parents(int s );

     void   EvolveSubPopulations(int repetitions,double h ,double mutation,int depth, ofstream &out2);

       };

void    CoEvolution:: InitializeSpecies(int popsize)
{         PopSize = popsize;

           GAvector SpeciesP(NoSpecies);
          Species = SpeciesP;

	 for( int Sp = 0; Sp < NoSpecies; Sp++){
	 	     NotConverged.push_back(false);
	 	     }




	 TotalSize = 0;
		for( int row = 0; row< NoSpecies ; row++)
			TotalSize+= SpeciesSize[row];

		for( int row = 0; row< TotalSize ; row++)
		        Individual.push_back(0);

	 for( int s =0; s < NoSpecies; s++){

	 	Species[s].NumVariable= SpeciesSize[s];
		  Species[s].InitilisePopulation(popsize);
     }



	        TableSp.resize(NoSpecies);

             for( int row = 0; row< NoSpecies ; row++)
	            for( int col = 0; col < SpeciesSize[row]; col++)
	         	   TableSp[row].SingleSp.push_back(0);


}


void    CoEvolution:: PrintSpecies()
{

	 for( int s =0; s < NoSpecies; s++){
		 Species[s].PrintPopulation();
		cout<<s<<endl;
     }

}

void    CoEvolution:: GetBestTable(int CurrentSp)
{
  int Best;

		 for(int sN = 0; sN < CurrentSp ; sN++){
		    Best= Species[sN].BestIndex;

          // cout<<Best<<endl;
		  for(int s = 0; s < SpeciesSize[sN] ; s++)
		    	 TableSp[sN].SingleSp[s] = Species[sN].Population[Best].Chrome[s];
		 }

		 for(int sN = CurrentSp; sN < NoSpecies ; sN++){
			// cout<<"g"<<endl;
			Best= Species[sN].BestIndex;
			   //cout<<Best<<" ****"<<endl;
	     for(int s = 0; s < SpeciesSize[sN] ; s++)
			TableSp[sN].SingleSp[s]= Species[sN].Population[Best].Chrome[s];
				 }

}

void   CoEvolution:: Join()
{


	int index = 0;

	  for( int row = 0; row< NoSpecies ; row++){
		for( int col = 0; col < SpeciesSize[row]; col++){

		  Individual[index] =  TableSp[row].SingleSp[col];
		  index++;
		}

	  }


}

void   CoEvolution:: Print()
{



	  for( int row = 0; row< NoSpecies ; row++){
		for( int col = 0; col < SpeciesSize[row]; col++){
		 cout<<TableSp[row].SingleSp[col]<<" ";   }
            cout<<endl;
	  }
            cout<<endl;

            for( int row = 0; row< TotalSize ; row++)
              cout<<Individual[row]<<" ";
            cout<<endl<<endl;
}

void    CoEvolution:: EvaluateSpecies( )
{


	 for( int SpNum =0; SpNum < NoSpecies; SpNum++){

		 GetBestTable(SpNum);

//---------make the first individual in the population the best

		 for(int i=0; i < Species[SpNum].NumVariable; i++)
		  TableSp[SpNum].SingleSp[i] = Species[SpNum].Population[0].Chrome[i];

		    Join();
		 Species[SpNum].Population[0].Fitness =  ObjectiveFunc(Individual);
		 TotalEval++;

		 Species[SpNum].BestFit = Species[SpNum].Population[0].Fitness;
		 Species[SpNum].BestIndex = 0;
		// cout<<"g"<<endl;
			 //------------do for the rest

		 for( int PIndex=0; PIndex< PopSize; PIndex++ ){


			 for(int i=0; i < Species[SpNum].NumVariable; i++)
	          TableSp[SpNum].SingleSp[i]  = Species[SpNum].Population[PIndex].Chrome[i];


		     Join();
             //Print();

		     Species[SpNum].Population[PIndex].Fitness = ObjectiveFunc(Individual);//
		     TotalEval++;

		     if ((MINIMIZE * Species[SpNum].BestFit) > (MINIMIZE * Species[SpNum].Population[PIndex].Fitness))
		       {
		    	 Species[SpNum].BestFit = Species[SpNum].Population[PIndex].Fitness;
		    	 Species[SpNum].BestIndex = PIndex;
		    	//  cout<<Species[SpNum].Population[PIndex].Fitness<<endl;
		       }

         }

			// cout<< Species[SpNum].BestIndex<<endl;
		cout<<SpNum<<" -- "<<endl;
     }







}

double   CoEvolution:: ObjectiveFunc(Layer x)
{


    int i,j,k;
  double fit, sumSCH;

  fit=0.0;

 if(ProbNum==1){
  // Ellipsoidal function
  for(j=0;j< x.size();j++)
    fit+=((j+1)*(x[j]*x[j]));
}

 else if(ProbNum==2){
  // Schwefel's function
  for(j=0; j< x.size(); j++)
    {
      for(i=0,sumSCH=0.0; i<j; i++)
	sumSCH += x[i];
      fit += sumSCH * sumSCH;
    }
}

 else if(ProbNum==3){
  // Rosenbrock's function
  for(j=0; j< x.size()-1; j++)
    fit += 100.0*(x[j]*x[j] - x[j+1])*(x[j]*x[j] - x[j+1]) + (x[j]-1.0)*(x[j]-1.0);
}

  return(fit);

  return(fit);
}


void CoEvolution::find_parents(int s )   //here the parents to be replaced are added to the temporary sub-population to assess their goodness against the new solutions formed which will be the basis of whether they should be kept or not
{
  int i,j,k;
  double u,v;

  Species[s].my_family();

  for(j=0;j<family;j++)
    {

    	  Species[s].NewPop[Species[s].kids+j].Chrome = Species[s].Population[Species[s].mom[j]].Chrome;



		 GetBestTable(s);
		 for(int i=0; i < Species[s].NumVariable; i++)
		  TableSp[s].SingleSp[i] =   Species[s].NewPop[Species[s].kids+j].Chrome[i];
		    Join();

		 Species[s].NewPop[Species[s].kids+j].Fitness = ObjectiveFunc(Individual);
		 TotalEval++;
    }
}


void CoEvolution::EvalNewPop(int pass, int s )
{

	 GetBestTable(s);
	 for(int i=0; i < Species[s].NumVariable; i++)
     TableSp[s].SingleSp[i] = 	Species[s].NewPop[pass].Chrome[i];
     Join();
  //   cout<<  Species[s].NewPop[pass].Fitness << "        pop fit"<<endl;

	 Species[s].NewPop[pass].Fitness =   ObjectiveFunc(Individual);
	 TotalEval++;


	  //   cout<<  Species[s].NewPop[pass].Fitness << "is new pop fit"<<endl;
}
void  CoEvolution::sort(int s)

{
  int i,j, temp;
  double dbest;

  for (i=0;i<(Species[s].kids+family);i++) Species[s].list[i] = i;

  if(MINIMIZE)
    for (i=0; i<(Species[s].kids+family-1); i++)
      {
	dbest = Species[s].NewPop[Species[s].list[i]].Fitness;
	for (j=i+1; j<(Species[s].kids+family); j++)
	  {
	    if(Species[s].NewPop[Species[s].list[j]].Fitness < dbest)
	      {
		dbest = Species[s].NewPop[Species[s].list[j]].Fitness;
		temp = Species[s].list[j];
		Species[s].list[j] = Species[s].list[i];
		Species[s].list[i] = temp;
	      }
	  }
      }
  else
    for (i=0; i<(Species[s].kids+family-1); i++)
      {
	dbest = Species[s].NewPop[Species[s].list[i]].Fitness;
	for (j=i+1; j<(Species[s].kids+family); j++)
	  {
	    if(Species[s].NewPop[Species[s].list[j]].Fitness > dbest)
	      {
		dbest = Species[s].NewPop[Species[s].list[j]].Fitness;
		temp = Species[s].list[j];
		Species[s].list[j] = Species[s].list[i];
		Species[s].list[i] = temp;
	      }
	  }
      }
}


void CoEvolution::rep_parents(int s )   //here the best (1 or 2) individuals replace the family of parents
{
  int i,j;
  for(j=0;j<family;j++)
    {

   Species[s].Population[Species[s].mom[j]].Chrome = Species[s].NewPop[Species[s].list[j]].Chrome;

    	  GetBestTable(s);

    		 for(int i=0; i < Species[s].NumVariable; i++)
    	 TableSp[s].SingleSp[i] =   Species[s].Population[Species[s].mom[j]].Chrome[i];
    	  Join();

    	  Species[s].Population[Species[s].mom[j]].Fitness =   ObjectiveFunc(Individual);
    	  TotalEval++;
    }
}

void CoEvolution:: EvolveSubPopulations(int repetitions,double h,   double mutation,int depth, ofstream & out1)
{
double tempfit;
		int count =0;
		int tag;
		kid = KIDS;
               int numspecies =0;




	for( int s =0; s < NoSpecies; s++) {

               for (int r = 0; r < repetitions; r++){  //evolve each sub-population for r generations

	    		tempfit=   Species[s].Population[Species[s].BestIndex].Fitness;
	    		    Species[s].kids = KIDS;


	    	 	Species[s].RandomParents();

	 	        for(int i=0;i<	Species[s].kids;i++)
	 	   	     {
	 	   	      tag = 	Species[s].GenerateNewPCX(i, mutation, depth); //generate a child using PCX

	 	   	        if (tag == 0) {
                                   NotConverged[s]=false;
                                 //NewPop[pass].Chrome[k] end = true;
                               //   out1<<"tag1"<<endl;
                                   break;
                              }
	 	   	     }
	 	   	          if (tag == 0) {
                                       //   end = true;
                                       // out1<<"tag2"<<endl;
	 	   	        	NotConverged[s]=false;
	 	   	          }

	 	   	     for(int i=0;i<	Species[s].kids;i++)
	 	   	      EvalNewPop(i,s );


	 	      find_parents(s );  // form a pool from which a solution is to be
	 	   	                           //   replaced by the created child

	 	  	    Species[s].sort();          // sort the kids+parents by fitness
            //       sort(s);
 	 	      rep_parents(s );   // a chosen parent is replaced by the child


	 	  	   Species[s].BestIndex=0;

	 	  	         tempfit= Species[s].Population[0].Fitness;

	 	  	         for(int i=1;i<PopSize;i++)
	 	   			    if((MINIMIZE *    Species[s].Population[i].Fitness) < (MINIMIZE * tempfit))
	 	   			      {
	 	   				tempfit= Species[s].Population[i].Fitness;
	 	   			   Species[s].BestIndex=i;
	 	   			      }


               }//r

	     	}//species
}





//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class CombinedEvolution        :    public    CoEvolution   //this class employs cooperative coevolution (CoEvolution ).
{


       public:
    int TotalEval;
     int TotalSize;
   double Energy;
   int SAFuncEval;
     double Train;
          double Test;
           double Error;
       	 CoEvolution cc; //declare the CC class that will be used


int Cycles;
        bool Sucess;

          CombinedEvolution(){

          }

          int GetEval(){
            return TotalEval;
                  }
          int GetCycle(){
                     return Cycles;
                           }
          double GetError(){
                    return Error;
                          }

          bool GetSucess(){
                              return Sucess;
                                    }

       void   Procedure(bool bp,   double h, ofstream &out1, ofstream &out2, ofstream &out3,double mutation,int depth );



};

void    CombinedEvolution:: Procedure(bool bp,   double h, ofstream &out1, ofstream &out2, ofstream &out3,double mutation,int depth )
{
        bool usememe = true;


  const int CCPOPSIZE = 100; //size of all the subpopulations

   const int maxgen =500000; //max number of function eval. (termination criteria)

      const int NumSubPop = 10; //number of sub-populations

      const int SubPopSize = 5; //size of each sub-population

   const double  MinError = 1E-20;//min error ( termination criteria )

	 const int NumGen = 1; //evolve each sub-population for NumGen generations



          Sucess= false;
//-----------------------------------------------------------
	  for(int n = 0; n < NumSubPop ; n++)
		  cc.SpeciesSize.push_back(SubPopSize);
            cc.ProbNum = (int)h;

	  cc.NoSpecies = NumSubPop;
      cc.InitializeSpecies(CCPOPSIZE);
          cc.EvaluateSpecies( );

      cout<<" Evaluated cc ----------->"<<endl;
    //-----------------------------------------------


      	  TotalEval=0;
      	  cc.TotalEval=0;

int count =10;


        while(TotalEval<=maxgen){

                   double ccError= 10000;
                   double ccTrain =0;

      	                   cc.EvolveSubPopulations(NumGen,0, 0,0, out2); //call CC to evolve each Subpop
      	                  cc.Join();
      	                  int bestindex =  cc.Species[cc.NoSpecies-1].BestIndex;
                          Error= cc.Species[cc.NoSpecies-1].Population[bestindex].Fitness;

           if(TotalEval%4000==0){
           out1<<  Error <<"   "<<TotalEval<<endl;
           cout<<  Error <<"   "<<TotalEval<<endl;}

                 TotalEval =   cc.TotalEval;


                          if(Error<MinError){

                          Sucess =true;
                          break;}






      	               	    count++;

      	               	       }










                 out2<<h<<"   "<<Error<<"   "<<TotalEval<<endl;


}













//---------------------------------------------------------------------------------------
int main(void)
{


	ofstream out1;
		out1.open("Oneout1.txt"); //output convergence for each run
	ofstream out2;
	     out2.open("Oneout2.txt"); //output the number of func eval for each run
	 	ofstream out3;
	 	     out3.open("Oneout3.txt"); //output mean and confidence intervals for func eval and error. also num of successful runs.




     for(double x=1;x<=1;x++){ //for the three func optimisation problems
    	 for(double y=1;y<= 1; y+=1){ //you can configure this to set your experiments -- (in case additional parameters needed)

     //do x number of experiments and calculate mean, confidence interval given the n number of successful runs...

    	 Sizes EvalAverage; //mean
    	 Layer ErrorAverage; //error
    	Layer CycleAverage; //function evals

    	 int MeanEval=0;
    	 double MeanError=0;
    	double MeanCycle=0;

    	 int EvalSum=0;
    	 double ErrorSum=0;
    	double CycleSum=0;
    	double maxrun =25; //number of experiments for each problem
int success =0;
    	 for(int run=1;run<=maxrun;run++){
    	  CombinedEvolution Combined;

          Combined.Procedure(false, x, out1, out2, out3, 0  , y );
if(Combined.GetSucess()){
   success++; //Count how many runs were sucessful, i.e, how many runs converged to the min. error before reaching max func. evals.
          }

//Include un-sucessful runs in the calculation of the mean and confidence intervals.
 //
  ErrorAverage.push_back(Combined.GetError());
    	  MeanError+= Combined.GetError();

    	  CycleAverage.push_back(Combined.GetCycle() );
    	      	  MeanCycle+= Combined.GetCycle();


          EvalAverage.push_back(Combined.GetEval());
          MeanEval+=Combined.GetEval();

    	               }//run

    	 MeanEval=MeanEval/EvalAverage.size();
    	 MeanError=MeanError/ErrorAverage.size();
    	 MeanCycle=MeanCycle/CycleAverage.size();

    	  for(int a=0; a < EvalAverage.size();a++)
    	EvalSum +=	(EvalAverage[a]-MeanEval)*(EvalAverage[a]-MeanEval);

    	  EvalSum=EvalSum/ EvalAverage.size();
    	  EvalSum = sqrt(EvalSum);

    	  EvalSum = 1.96*(EvalSum/sqrt( EvalAverage.size()));
    	  for(int a=0; a < CycleAverage.size();a++)
    		  CycleSum +=	(CycleAverage[a]-MeanCycle)*(CycleAverage[a]-MeanCycle);

    	  CycleSum=CycleSum/ CycleAverage.size();
    	  CycleSum = sqrt(CycleSum);
    	  CycleSum = 1.96*(CycleSum/sqrt( CycleAverage.size()));
    	  for(int a=0; a < ErrorAverage.size();a++)
        ErrorSum +=	(ErrorAverage[a]-MeanError)*(ErrorAverage[a]-MeanError);

    	 ErrorSum=ErrorSum/ ErrorAverage.size();
    	ErrorSum = sqrt(ErrorSum);
        ErrorSum = 1.96*(ErrorSum/sqrt(ErrorAverage.size()) );
 	out3<<y<<" "<<x<<"  "<<MeanEval<<"  "<<EvalSum<<"  "<<MeanError<<"  "<<ErrorSum<<" "<<MeanCycle<<"  "<<CycleSum<<"  "<<success<<"  "<<endl;




     	EvalAverage.empty();
     	ErrorAverage.empty();
     	CycleAverage.empty();
    	 }
     }
     out3<<"\\hline"<<endl;

	out1.close();
	out2.close();
	out3.close();


 return 0;

};


