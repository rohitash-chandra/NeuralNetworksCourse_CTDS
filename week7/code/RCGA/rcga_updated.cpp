/*
*	Created by: Dr. Rohitash Chandra
*	Documented by: Shelvin Chand 
*	Software Foundation Fiji (Artficial Intelligence and Cybernetics Research Group--www.softwarefoundationfiji.org/aicrg)
*	This code is being released under the Creative Common License 3.0

/*
	--Genectic Algorithm--	


    Problem No. 3 is Forward Kinematics Problem (6RPR) 
*/

#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <ctime>
#include <sys/time.h>  
#include <stdio.h>
#include <time.h>

time_t TicTime;
time_t TocTime;

using namespace::std;
//type delcalaration
typedef vector<double> Nodes;
typedef vector<double> Layer;
typedef vector<int> Sizes;
typedef vector<vector<double> > Weight;
typedef vector<vector<double> > Data;
  
class GeneticAlgorithmn{
       //class variables
       protected:
            Data Chromozone;
			Data FinalChromozone;
			Data FirstChromozone;
			Data NewGeneration;
			Layer Fitness;
			Layer FitnessRatio;   
			Data Distance;
			
			int Population;
			int Stringsize;
			int chrome;       	 
			int Time;
			int function; 
			
			    int TotalEval;

              double Error;

             double Cycles;
           bool Success;
       //class methods
       public:
			GeneticAlgorithmn(int population,int stringSize, int func)
			{//constructor
				chrome = 0;
				Population = population;//population size
				Stringsize = stringSize;//chromozone size
				function=func;//function to evaluate
			}    
      
			double Random() ; 
			
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
                                  return Success;
                                        } 
		 
       
			double FitnessFunc(Layer x, int ProbNum);
			
			int  Algorithmn(double Crossover,double Mutation, ofstream &output1, ofstream &output2);   
			
			double Evaluate()  ;
   
			int Select() ;
   
			void CrossoverAndMutate(int leftpair,int rightpair,double Crossover,double Mutation,int position);
      
			void InitilisePopulation(int population, int stringsize );
    
			double MaxFitness(); 
    
			void Print()  ;
   
			int MaxLocation();
  
			double RandomWeights();

			double RandomAddition();
    		 
};
   

double GeneticAlgorithmn::Random()
{     
    double string;
    return string = rand()%2;    
     
}

double GeneticAlgorithmn::RandomWeights()
{     
    int chance;
    double randomWeight;
    double NegativeWeight;
    chance =rand()%2;//randomise between negative and positive numbers
      
    if(chance ==0){
		return (((rand()%10000)*0.0001) * 5);
    }
     
    if(chance ==1){
		return ((rand()%10000)*0.0001) * -5;
	}
     
}

double GeneticAlgorithmn::RandomAddition()
{     
    int chance;
    double randomWeight;
    double NegativeWeight;
    chance =rand()%2;//randomise between negative and positive numbers
      
    if(chance ==0){ 
		return drand48()/10000;
    }
     
    if(chance ==1){
     
		return -drand48()/10000;
    }
     
}
 

void GeneticAlgorithmn::InitilisePopulation(int population, int stringsize )
{
    int count = 0;

	for(int r=0; r <  population  ; r++)
		Chromozone.push_back(vector<double> ());//create matrix
   
	for(int row = 0; row < population  ; row++) { 
		for(int col = 0; col < stringsize ; col++) 
			Chromozone[row].push_back(RandomWeights());//initialize with randome weights
	}

	for(int r=0; r <  population  ; r++)
		FinalChromozone.push_back(vector<double> ());//create matrix
   
	for(int row = 0; row < population  ; row++) { 
		for(int col = 0; col < stringsize ; col++) 
			FinalChromozone[row].push_back(0);//initialise with 0s for each row
	}
  
    for(int r=0; r <  population  ; r++)
		FirstChromozone.push_back(vector<double> ());//create matrix
   
	for(int row = 0; row < population  ; row++) { 
		for(int col = 0; col < stringsize ; col++) 
			FirstChromozone[row].push_back(RandomWeights());//intialize with random weights
	}
 
    for(int r=0; r <  population  ; r++)
		NewGeneration.push_back(vector<double> ());//create matrix to hold data
   
	for(int row = 0; row < population  ; row++) { 
		for(int col = 0; col < stringsize ; col++) 
			NewGeneration[row].push_back(0);//intialize newgeneration vector with 0s for all rows
	}
    
	for(int r=0; r <  population  ; r++)
		Fitness.push_back(0);//initialize fitness vector with 0s
         
    for(int r=0; r <  population  ; r++)
		FitnessRatio.push_back(0);  //intialize fitness_ratio vector with 0s 
      
}

 

double GeneticAlgorithmn::FitnessFunc(Layer x, int ProbNum)
{

	int i,j,k;
    double z; 
	double fit = 0.0;  
	double   sumSCH; 

	if(ProbNum==1){
	// Ellipsoidal function
		for(j=0;j< x.size();j++)
			fit+=((j+1)*(x[j]*x[j]));
	}
	else if(ProbNum==2){
		// Schwefel's function
		for(j=0; j< x.size(); j++)
		{
			sumSCH=0;
			for(i=0; i<j; i++)
			sumSCH += x[i];
			fit += sumSCH * sumSCH;
		}
	}
	else if(ProbNum==3){



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
pow(sqrt(pow(x3-x2,2)+pow(y3-y2,2)+pow(z3-z2,2)) - sqrt(9349310123.0/500000.0),2);
	}
	return  1/fit;
}

/*
	--Print--
	Output fitness, fitness ratio and chromozone values
*/ 
void GeneticAlgorithmn::Print()
{
	
	for(int row = 0; row < Population  ; row++) { 
		for(int col = 0; col < Stringsize ; col++) 
			cout<<Chromozone[row][col]<<" ";//output all the values of the population
			cout<<"       "<<1.0/ FitnessFunc(Chromozone[row],function );
			cout<<endl;
	}
   
	cout<<endl; 
	cout<<endl;
	cout<<endl;
	cout<<"-----------------------------------------"<<endl;
	cout<<" Fitness: "<<endl<<endl;
    for(int r=0; r <  Population  ; r++)
		cout<<Fitness[r]<<" ";//output the fitness for the entire population
		cout<<endl;   
	
	cout<<" Fitness Ratio: "<<endl<<endl;
    for(int r=0; r <  Population  ; r++)
		cout<<FitnessRatio[r]<<" ";  //output the fitness ration for the entire ratio
		cout<<endl;         
               
               
}

double GeneticAlgorithmn::Evaluate() 
{
    double sum =0;
           
    for(int label = 0; label < Population  ; label++) {           
        Fitness[label] =    FitnessFunc(Chromozone[label], function );
  
        sum+=Fitness[label];
	}
          
    for(int label = 0; label < Population  ; label++) { 
        FitnessRatio[label] = Fitness[label]/sum*100;
	}
                  
}      
      

 double GeneticAlgorithmn:: MaxFitness(){
    double max = 0;
        
    for(int label = 0; label <Population  ; label++) { 
        if( (Fitness[label]) >   max ){ 
			max = Fitness[label]; 
			
		}
  
    }
        
    return max;
}


int GeneticAlgorithmn:: MaxLocation(){
        
    for(int label = 0; label <Population  ; label++) { 
        if( MaxFitness() ==Fitness[label]  ) 
			return label; 
    }
        
        
    return 0;
}    

int GeneticAlgorithmn:: Select()
{
    Nodes Wheel(Population+1);
    double random;
    random = rand()%100;
    double sum = 0;   
    Wheel[0] = 0;
    for(int label = 1; label < Wheel.size()  ; label++) { 
        Wheel[label]  =    (FitnessRatio[label-1]+Wheel[label-1]);
	
	}
         
	for(int label = 0; label <(Wheel.size()-1 ) ; label++)  {
        if( (random >=Wheel[label])&&(random < Wheel[label+1])) {  
           
			return label;
        }
	}
}
  

void GeneticAlgorithmn:: CrossoverAndMutate(int leftpair,int rightpair,double Crossover,double Mutation, int position)
{
                 
	Layer LeftChromozone(Stringsize);                           
	Layer RightChromozone(Stringsize);
	Nodes ChildChromozone(Stringsize);  
	Nodes Temporary(Stringsize);
	
	double leftfit = 0;
	double rightfit = 0;
	double childfit = 0;
	double mutatechildfit = 0;
   
	for(int gene = 0; gene < Stringsize ; gene++){
		LeftChromozone[gene] = Chromozone[leftpair][gene];
		RightChromozone[gene] = Chromozone[rightpair][gene];
	}

	leftfit = Fitness[leftpair];
	rightfit = Fitness[rightpair];


	double random = rand() % 100 ;
	int chooseparent  ;
	int choosegene  ;
	bool cross = false;      
	double newfit = 0;
	double  Lfit = 0;
	double Rfit = 0;
	
	if( random < (100 *Crossover)){
	
		cross = true;
		double randomalpha = drand48();
		Lfit = FitnessFunc(LeftChromozone, function) ;
		Rfit = FitnessFunc(RightChromozone, function) ;
		
		//Wrights Heuristic Crossover
		if ( Lfit > Rfit){
			for(int gene = 0; gene < Stringsize ; gene++)
				ChildChromozone[gene] = randomalpha *((LeftChromozone[gene]-RightChromozone[gene]))+ LeftChromozone[gene];
		}else{
			for(int gene = 0; gene < Stringsize ; gene++)
				ChildChromozone[gene] = randomalpha *((RightChromozone[gene]-LeftChromozone[gene]))+ RightChromozone[gene];   
		}

	}
	else{ 
		ChildChromozone= LeftChromozone; 

	}
	int ch = rand()%100;

	
	double randommutate;
	int seed = rand()%Stringsize;
	double MutationNonUniform = 0.1;
	int chance = rand()%1000;
	double Max = 0;
    int newseed = rand()%Stringsize;
	
	
	if( chance < (1000 *MutationNonUniform)){ 
		
		for(int gene = 0 ; gene < Stringsize ; gene++)
			ChildChromozone[ gene ] +=  RandomAddition();
      
    } 
      
    for(int gene = 0; gene < Stringsize ; gene++){
		NewGeneration[position][gene] = ChildChromozone[gene]; 
    }  
}
 
     
int GeneticAlgorithmn:: Algorithmn(double Crossover,double Mutation, ofstream &output1, ofstream &output2 )   
{
	
	
	clock_t start = clock();
    int gene = 1;
	double maxfitness = 0;                    
	InitilisePopulation( Population, Stringsize);                   
 					 
	Evaluate(); 
	maxfitness = MaxFitness();
 
	//cout<<"Max Fitness:"<< maxfitness<<endl;	
	//Success = false;
   
	int leftpair = 0;
	int rightpair = 0; 
	bool stop = false;
	double train = 0;    
	int generation = 0;  
	int w=0;	
	
	
Success = false;

	//while(w<=1){	
	while((gene*Population)< 5000000){      
			 
		Evaluate();                     
		maxfitness =  MaxFitness(); 

		chrome = 0;
		int count =0;
		
		while((count) < Population ) {  
			do{
				leftpair = Select() ;
				rightpair = Select() ;
			} while (leftpair == rightpair);
	   
			CrossoverAndMutate(leftpair,rightpair,Crossover, Mutation,count);
   
			count ++; 
		  
		}
		count=0;
		
		 
			
		Chromozone = NewGeneration;
		//cout<<"Max Fitness: "<<1/maxfitness <<" " <<gene<<endl;
 
		 Error = 1/maxfitness;
		 if (Error < 1E-20){
			 Success = true;
			 break;
			 
			 
		 }
		 
		 if (gene%1000)
		 cout<<Error<<" "<<gene<<endl;
		gene++;

	}
	

   clock_t finish = clock();

   Cycles = ((double)(finish - start))/CLOCKS_PER_SEC;
cout<<Cycles<<" ----"<<endl;


   TotalEval = gene * Population;
   
	output1<<1/maxfitness <<" " <<gene<<endl;
		for(int x=0; x<Stringsize; x++){		
			output2<<Chromozone[MaxLocation()][x]<< " ";
		}
		output2<<endl;
		
	output2<<"   ---    "<<1/maxfitness <<" " <<
   TotalEval<< "    "<<  Cycles <<endl;
  

    return gene;                       
}
  
  
  



int main(void)
{

  int generation = 0; 
	int function=1; //FKP problem (robotics)
    double Threshold = 2 ; 
    double Crossover = 0.9; 
    int Dimension = 9; // (robotics FKP problem)
      
 srand (time(NULL));

 ofstream out1;
		out1.open("out1.txt");
	ofstream out2;
	     out2.open("out2.txt");
	 	ofstream out3;
	 	     out3.open("out3.txt");

	 	   
       	


	//int Population = 200;
	double Mutation = 0.1;
	
	for(int Population =  150; Population<=400; Population+=50){
		
		 int MeanEval=0;
	 	       	 double MeanError=0;
	 	       double MeanTime=0;

	 	       	 int FuncEvalCI=0;
	 	       	 double ErrorCI=0;
	 	       	double TimeCI=0;
	 	       	
                   int CountSuccess = 0;
		 Sizes EvalAverage;
	 	       	 Layer ErrorAverage;
	 	       	 Layer TimeAverage; 
	 	       	 
		int maxRun = 5;// num of exp


		for( int run =1 ; run <= maxRun; run ++){
          		
			GeneticAlgorithmn ga(Population,Dimension, function);  
			
     		generation = ga.Algorithmn(Crossover, Mutation, out1 ,out2);
      
      if (ga.GetSucess()){

             
                   CountSuccess++;
		  
              EvalAverage.push_back(ga.GetEval());
	                 MeanEval+=ga.GetEval();

	                 ErrorAverage.push_back(ga.GetError());
	           	  MeanError+= ga.GetError();
	           	  
	           	  TimeAverage.push_back(ga.GetCycle());
	           	      	  MeanTime+= ga.GetCycle();
                       }
			
         
     
	}
	
	MeanEval=MeanEval/CountSuccess;
	MeanError=MeanError/CountSuccess;
	MeanTime=MeanTime/CountSuccess;

	   for(int a=0; a < EvalAverage.size();a++)
	FuncEvalCI +=	(EvalAverage[a]-MeanEval)*(EvalAverage[a]-MeanEval);

	  FuncEvalCI=FuncEvalCI/ EvalAverage.size();
	  FuncEvalCI = sqrt(FuncEvalCI);
	  
	 FuncEvalCI = 1.96*(FuncEvalCI/sqrt( EvalAverage.size()));

	  for(int a=0; a < TimeAverage.size();a++)
		  TimeCI +=	(TimeAverage[a]-MeanTime)*(TimeAverage[a]-MeanTime);

	  TimeCI=TimeCI/ TimeAverage.size();
	  TimeCI = sqrt(TimeCI);
	  TimeCI = 1.96*(TimeCI/sqrt(TimeAverage.size()));
	  for(int a=0; a < ErrorAverage.size();a++)
   ErrorCI +=	(ErrorAverage[a]-MeanError)*(ErrorAverage[a]-MeanError);

	 ErrorCI=ErrorCI/ ErrorAverage.size();
	ErrorCI = sqrt(ErrorCI);
   ErrorCI = 1.96*(ErrorCI/sqrt(   ErrorAverage.size()));  
	out3<< "   "<<MeanEval<<" "<<FuncEvalCI<<"         "<<MeanError<<" "<<ErrorCI<<"      "<<MeanTime<<"  "<<TimeCI<<"      "<<Population<<"   "<<ErrorAverage.size()<<endl;
	EvalAverage.empty(); 
	 ErrorAverage.empty(); 
	 TimeAverage.empty(); 
	         
	
}
 
	out1.close();
	out2.close();
	out3.close();

cout<<"hello world"<<endl;
 
 return 0;

} 


