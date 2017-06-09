/*
 *      Created on: 2005/2006
 *      Author: Rohitash Chandra 


        Simple FNN with Vanilla BP. Weight decay has been used. 

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

//Type declaratiobns
typedef vector<double> Layer;//for the different layers within the neural network
typedef vector<double> Nodes;//for the nodes within the network-- a vector matrix of type double.
typedef vector<int> Sizes; // Vector of Int
typedef vector<double> Frame;
typedef vector<vector<double> > Weight;//for managing the weights of the neural network connections-=-a vector matrix of type double
typedef vector<vector<double> > Data;
 

const int LayersNumber = 3; //total number of layers.  Can be more if you wish to have mroe than one hidden layer. Need to change topology in main if increase more than 1 hidden layer. i.e layer is 4, topology would be [input, hidden, hidden, output]

  

//-----------------------
const double alpha = 0.00001; //weight decay constant

const double error_tolerance=0.2;//amount of error accepted during training  

const int MaxTime  = 2000; // Epochs

const double MinTrain = 95.00; //stop if this value is reached in training performance
 



class Layers{  

    friend class NeuralNetwork;

    protected:
		//class variables
        Weight Weights ;//neural network weights 
        Weight WeightChange;//change in weight after each iteration
        Weight H;
        Layer Outputlayer;
        Layer Bias;//keeping track of the bias factor in the network
        Layer B;
        Layer Gates;
        Layer BiasChange;//keeping track of change in Bias factor
        Layer Error;//error after each iteration

    public:
		//functions
		Layers()
        {
			//contructor method
        }

};

class TrainingExamples{
    friend class NeuralNetwork;
    
	protected:
       //class variables
	   Data  InputValues;
       	   Data  DataSet;
           Data  OutputValues;
	   char* FileName;
	   int Datapoints;
           int colSize ;
           int inputcolumnSize ;
           int outputcolumnSize ;
	   int datafilesize  ;
    
	public:
		//function declarations
		TrainingExamples()
		{
			//constructor
		} ;
		//overriden constructor
		TrainingExamples( char* File, int size, int length, int inputsize, int outputsize ){
			//initialize functions and class variables
			inputcolumnSize = inputsize ;
			outputcolumnSize = outputsize;
			datafilesize = inputsize+ outputsize;
			colSize = length;
			Datapoints= size;

			FileName = File;
			InitialiseData();
		}
		
		void printData();
		
		void InitialiseData();

};

void TrainingExamples:: InitialiseData()
{
	ifstream in( FileName );
        if(!in) {
		cout << endl << "failed to open file" << endl;//error message for reading from file
        }

	//initialise dataset vectors
	for(int r=0; r <  Datapoints ; r++)
	DataSet.push_back(vector<double> ());

	for(int row = 0; row < Datapoints ; row++) {
		for(int col = 0; col < colSize ; col++)
			DataSet[row].push_back(0);
	}
      // cout<<"printing..."<<endl;
    for(  int row  = 0; row   < Datapoints ; row++)
    for( int col  = 0; col  < colSize; col ++)
      in>>DataSet[row ][col];
      //-------------------------
    //initialise intput vectors
	for(int  r=0; r < Datapoints; r++)
		InputValues.push_back(vector<double> ());

    for(int  row = 0; row < Datapoints ; row++)
		for(int col = 0; col < inputcolumnSize ; col++)
			InputValues[row].push_back(0);//initialise with 0s

    for(int  row = 0; row < Datapoints ; row++)
		for(int col = 0; col < inputcolumnSize ;col++)
			InputValues[row][col] = DataSet[row ][col] ;//read values from the dataset vector
    
	//initialise output vectors
	for(int r=0; r < Datapoints; r++)
		OutputValues.push_back(vector<double> ());

    for( int row = 0; row < Datapoints ; row++)
		for( int col = 0; col < outputcolumnSize; col++)
			OutputValues[row].push_back(0);//initialse with 0s

    for( int row = 0; row < Datapoints ; row++)
		for(int  col = 0; col <  outputcolumnSize;col++)
			OutputValues[row][col]= DataSet[row ][ col +inputcolumnSize ] ;

    in.close();//close connection
 } 

void TrainingExamples:: printData()
{
    cout<<"printing...."<<endl;
	cout<<"Entire Data Set.."<<endl;
	for(int row = 0; row < Datapoints ; row++) {
		for(int col = 0; col < colSize ; col++)
			cout<<  DataSet[row][col]<<" ";//output entire set
			cout<<endl;
	}
    
	cout<<endl<<"Input Values.."<<endl;  
    
	for(int  row = 0; row < Datapoints ; row++) {
		for( int col = 0; col < inputcolumnSize; col++)
			cout<<  InputValues[row][col]<<" ";//output only input values
			cout<<endl;
    }
    
	cout<<endl<<"Expected Output Values.."<<endl;   
   
	for( int row = 0; row < Datapoints ; row++)  {
		for( int col = 0; col <  outputcolumnSize;col++)
			cout<< OutputValues[row][col] <<" " ;//print output values
			cout<<endl;
	}
}




class NeuralNetwork{ 
	
    protected:
		//class variables
		Layers nLayer[LayersNumber];
		double Heuristic;
		int StringSize;
		Layer ChromeNeuron;
		int NumEval;
		Data Output;
                Sizes layersize;
		double NMSE;
		
     public:
		//function declaration 
		NeuralNetwork( )
		{
			//constructor
		}
		
		NeuralNetwork(Sizes layer )
		{
			layersize  = layer;
			StringSize = (layer[0]*layer[1])+(layer[1]*layer[2]) +   (layer[1] + layer[2]);
		}

		void PlaceHeuristic(double H)
		{
			Heuristic = H;
        	}
		
	        //Some functions   are not currently used. You can use them depending on your requirements.
		double Random();
		
		double Sigmoid(double ForwardOutput);
		
		double NMSError() {return NMSE;} // not used - good for time series problems
		 
		
		void CreateNetwork(Sizes Layersize ,TrainingExamples TraineeSamples);
		
		void ForwardPass(TrainingExamples TraineeSamp,int patternNum,Sizes Layersize);
		
		void BackwardPass(TrainingExamples TraineeSamp,double LearningRate,int patternNum,Sizes Layersize);
		
		void PrintWeights(Sizes Layersize);// print  all weights
		
		bool ErrorTolerance(TrainingExamples TraineeSamples,Sizes Layersize, double TrainStopPercent);
		
	 	
		double SumSquaredError(TrainingExamples TraineeSamples,Sizes Layersize);
		
		int BackPropogation(  TrainingExamples TraineeSamples, double LearningRate,Sizes Layersize,char* Savefile,bool load);
		
		void SaveLearnedData(Sizes Layersize,char* filename) ;
 

		void LoadSavedData(Sizes Layersize,char* filename) ;

		double TestLearnedData(Sizes Layersize,char* filename,int  size, char* load,  int inputsize, int outputsize );

		double  CountLearningData(TrainingExamples TraineeSamples,int temp,Sizes Layersize);

        	double  TestTrainingData(Sizes Layersize, char* filename,int  size, char* load,  int inputsize, int outputsize , ofstream & out2 );

		double CountTestingData(TrainingExamples TraineeSamples,int temp,Sizes Layersize);
 
		bool CheckOutput(TrainingExamples TraineeSamples,int pattern,Sizes Layersize);
};
	
/*
	--Random--
	Is used to generate random numbers which are used to initialize weights and neurons when the network is created
*/
double NeuralNetwork::Random()//method for assigning random weights to Neural Network connections
{     
	int chance;
    double randomWeight;
    double NegativeWeight;
    chance =rand()%2;//randomise between positive and negative 

    if(chance ==0){
		randomWeight =rand()% 100;
		return randomWeight*0.05;//assign positive weight
    }

    if(chance ==1){
		NegativeWeight =rand()% 100;
		return NegativeWeight*-0.05;//assign negative weight
    }

}
/*
	--Sigmoid--
	Function to convert weighted_sum into a value between  -1 and 1
*/
double NeuralNetwork::Sigmoid(double ForwardOutput) 
{
      
    return  (1.0 / (1.0 + exp(-1.0 * (ForwardOutput) ) ));
}


void NeuralNetwork::CreateNetwork(Sizes Layersize,TrainingExamples TraineeSamples)//create network and initialize the weights
{
       

	int end = Layersize.size() - 1;

	for(int layer=0; layer < Layersize.size()-1; layer++){//go through each layer
  
        for(int  r=0; r < Layersize[layer]; r++)
			nLayer[layer].Weights.push_back(vector<double> ());  

        for( int row = 0; row< Layersize[layer] ; row++)
			for( int col = 0; col < Layersize[layer+1]; col++)
				nLayer[layer].Weights[row].push_back(Random());
     
		for(int  r=0; r < Layersize[layer]; r++)              
			nLayer[layer].WeightChange.push_back(vector<double> ());
                                                                   
        for( int row = 0; row < Layersize[layer] ; row ++)
			for( int col = 0; col < Layersize[layer+1]; col++)
				nLayer[layer].WeightChange[row ].push_back(0);

        for( int r=0; r < Layersize[layer]; r++)
			nLayer[layer].H.push_back(vector<double> ());//create matrix

        for(int  row = 0; row < Layersize[layer] ; row ++)
			for( int col = 0; col < Layersize[layer+1]; col++)
				nLayer[layer].H[row ].push_back(0);//initialize all the elements in H with 0 for each layer
     
    }

    for( int layer=0; layer < Layersize.size(); layer++){
		
		for( int row = 0; row < Layersize[layer] ; row ++)
			nLayer[layer].Outputlayer.push_back(0);//initialize neurons of each layer with 0s

		for( int row = 0; row < Layersize[layer] ; row ++)
			nLayer[layer ].Bias.push_back(Random());//the bias for each each layer and connection will be a random value

		for( int row = 0; row < Layersize[layer] ; row ++)
			nLayer[layer ].Gates.push_back(0);//initialize gates vector with 0s


		for( int row = 0; row < Layersize[layer] ; row ++)
			nLayer[layer ].B.push_back(0);//initialize with 0s

		for( int row = 0; row < Layersize[layer] ; row ++)//for each connection we will also keep track of change in bias factor
			nLayer[layer ].BiasChange.push_back(0);//initially it will be all 0

		for( int row = 0; row < Layersize[layer] ; row ++)
			nLayer[layer ].Error.push_back(0);// intialize error vector for each layer with 0s
    }
    
    for(int  r=0; r < TraineeSamples.Datapoints; r++)
        Output.push_back(vector<double> ());
    
	for( int row = 0; row< TraineeSamples.Datapoints ; row++)
        for(int  col = 0; col < Layersize[end]; col++)
			Output[row].push_back(0);// intialize all the rows in the output vector with 0s 

}


void NeuralNetwork::ForwardPass(TrainingExamples TraineeSamples,int patternNum,Sizes Layersize)
{
   //declaring essential variables
	double WeightedSum = 0;
    double ForwardOutput;//to hold output value between -1 and 1
	int end = Layersize.size() - 1; //know the last layer

    for(int row = 0; row < Layersize[0] ; row ++)
		nLayer[0].Outputlayer[row] = TraineeSamples.InputValues[patternNum][row];


    for(int layer=0; layer < Layersize.size()-1; layer++){
		for(int y = 0; y< Layersize[layer+1]; y++) {
			for(int x = 0; x< Layersize[layer] ; x++){
				WeightedSum += (nLayer[layer].Outputlayer[x] * nLayer[layer].Weights[x][y]);
                 }
				ForwardOutput = WeightedSum - nLayer[layer+1].Bias[y]; 
			//}

			nLayer[layer+1].Outputlayer[y] = Sigmoid(ForwardOutput);//convert the weighted sum to a value between 1 and -1 which will be the new value in the neuron

			WeightedSum = 0;
		}
		WeightedSum = 0;
	}//end layer

   //--------------------------------------------
   for(int output= 0; output < Layersize[end] ; output ++){
		Output[patternNum][output] = nLayer[end].Outputlayer[output];
  
	}

 }
 

void NeuralNetwork::BackwardPass(TrainingExamples TraineeSamp,double LearningRate,int patternNum,Sizes Layersize)	
{
	int end = Layersize.size() - 1;// know the end layer
    double temp = 0;

    // compute error gradient for output neurons
	for(int output=0; output < Layersize[end]; output++) {
		nLayer[end].Error[output] = (Output[patternNum][output]*(1-Output[patternNum][output]))*(TraineeSamp.OutputValues[patternNum][output]-Output[patternNum][output]);
    }
    //----------------------------------------
    
	for(int layer = Layersize.size()-2; layer != 0; layer--){

		for( int x = 0; x< Layersize[layer] ; x++){  //inner layer
			for(int  y = 0; y< Layersize[layer+1]; y++) { //outer layer
				temp += ( nLayer[layer+1].Error[y] * nLayer[layer].Weights[x][y]);
            }
			nLayer[layer].Error[x] = nLayer[layer].Outputlayer[x] * (1-nLayer[layer].Outputlayer[x]) * temp;
 
			temp = 0.0;//reset temp for the next neuron
		}
		temp = 0.0; //reset temp for the next layer

	}
   
  	double tmp;
  	//int layer =0;
	for( int layer = Layersize.size()-2; layer != -1; layer--){//go through all layers
 
		for( int x = 0; x< Layersize[layer] ; x++){  //inner layer
			for( int y = 0; y< Layersize[layer+1]; y++) { //outer layer
				tmp = (( LearningRate * nLayer[layer+1].Error[y] * nLayer[layer].Outputlayer[x])  );
				nLayer[layer].Weights[x][y] += ( tmp  -  ( alpha * tmp) ) ;//update weight
    
            }
		}
	}

   double tmp1;

    for( int layer = Layersize.size()-1; layer != 0; layer--){//go through all layers
     
        for( int y = 0; y< Layersize[layer]; y++){
			tmp1 = (( -1 * LearningRate * nLayer[layer].Error[y])  );//calculate change in bias
			nLayer[layer].Bias[y] +=  ( tmp1 - (alpha * tmp1))  ;//updated bias of layer
        
        }
	}
  

 }


bool NeuralNetwork::ErrorTolerance(TrainingExamples TraineeSamples,Sizes Layersize, double TrainStopPercent)
{   
	//declare essential variables
    double count = 0;
    int total = TraineeSamples.Datapoints;
    double accepted = total;
    double desiredoutput;
    double actualoutput;
    double Error;
    int end = Layersize.size() - 1;
	
	//go through all training samples
	for(int pattern = 0; pattern< TraineeSamples.Datapoints; pattern++){

		Layer Desired;
		Layer Actual;

		for(int i = 0; i <  Layersize[end] ;i++)
			Desired.push_back(0);//initialize vector for desired output with 0s
		for(int j = 0; j <  Layersize[end] ;j++)
			Actual.push_back(0);//initialize vector for actual output with 0s



		for(int output = 0; output < Layersize[end]; output++) {
			desiredoutput = TraineeSamples.OutputValues[pattern][output];
			actualoutput = Output[pattern][output];

			Desired[output] = desiredoutput;
			
			if((actualoutput >= 0)&&(actualoutput <= 0.2))
				actualoutput = 0;//round down
			else if((actualoutput <= 1)&&(actualoutput >= 0.8))
				actualoutput = 1;//round up

			Actual[output] =  actualoutput;
		}
		int confirm = 0;
    
		for(int b = 0; b <  Layersize[end] ;b++){
			if(Desired[b]== Actual[b] )
				confirm++;
                
			if(confirm == Layersize[end])
				count++;
				confirm = 0;//reset for next layer
       
		}
	}
    if(count ==accepted)
			return false;


	return true;
	
}

 
double NeuralNetwork::SumSquaredError(TrainingExamples TraineeSamples,Sizes Layersize)
{   int end = Layersize.size() - 1;//know last layer
    double Sum = 0;
    double Error=0;
    double ErrorSquared = 0;
    for(int pattern = 0; pattern< TraineeSamples.Datapoints ; pattern++){
		for(int output = 0; output < Layersize[end]; output++) {
			Error = fabs(TraineeSamples.OutputValues[pattern][output]) - fabs(Output[pattern][output]);
			ErrorSquared += (Error * Error);//square the error
		}

        Sum += (ErrorSquared);//add to cumulative error
        ErrorSquared = 0;//set error squared variable to 0 for next layer
  
	}
	return sqrt(Sum/TraineeSamples.Datapoints*Layersize[end]);//return square root of sum / (no. of training samples * no. of neurons in output layer)
}
 
void NeuralNetwork::PrintWeights(Sizes Layersize)//output the values of all the connection weights
{
    int end = Layersize.size() - 1;

    for(int layer=0; layer < Layersize.size()-1; layer++){

		cout<<layer<<"  Weights::"<<endl<<endl;
		for(int row  = 0; row <Layersize[layer] ; row ++){
			for(int col = 0; col < Layersize[layer+1]; col++)
				cout<<nLayer[layer].Weights[row ][col]<<" "; //output all values from the weights matrix for all layers
				cout<<endl;
        }
		cout<<endl<<layer<<"  WeightsChange::"<<endl<<endl;

		for( int row  = 0; row <Layersize[layer] ; row ++){
			for( int col = 0; col < Layersize[layer+1]; col++)
				cout<<nLayer[layer].WeightChange[row ][col]<<" ";//output all values from the weightchange matrix for all layers
				cout<<endl;
        }
		
		cout<<"--------------"<<endl;
	}

	for(int layer=0; layer < Layersize.size() ; layer++){
		cout<<endl<<layer<<"  Outputlayer::"<<endl<<endl;//output values from outputlayer
		for( int row = 0; row < Layersize[layer] ; row ++)
			cout<<nLayer[layer].Outputlayer[row] <<" ";
	
	cout<<endl<<layer<<"  Bias::"<<endl<<endl;
	for( int row = 0; row < Layersize[layer] ; row ++)
        cout<<nLayer[layer].Bias[row] <<" ";//output values from the Bias Matrix for each layer
	
	cout<<endl<<layer<<"  Error::"<<endl<<endl;
	for(int  row = 0; row < Layersize[layer] ; row ++)
        cout<<nLayer[layer].Error[row] <<" "; //output values from the error matrix for each layer
	
	cout<<"----------------"<<endl;

}

     }
/*
	--SaveLearnedData--
	Save the network weights and bias values which were able to achieve optimal results to file
*/
void NeuralNetwork::SaveLearnedData(Sizes Layersize, char* filename)//save data to file
{

	ofstream out;
	out.open(filename);
	if(!out) {
		cout << endl << "failed to save file" << endl;//error in writing to file
		return;
    }
	
    for(int layer=0; layer < Layersize.size()-1; layer++){//ouput weights
        for(int row  = 0; row <Layersize[layer] ; row ++){
			for(int col = 0; col < Layersize[layer+1]; col++)
				out<<nLayer[layer].Weights[row ][col]<<" ";
				out<<endl;
        }
        out<<endl;//blank line
    }
 
  // output bias.
	for(int  layer=1; layer < Layersize.size(); layer++){
		for(int y = 0 ; y < Layersize[layer]; y++) {
			out<<	nLayer[layer].Bias[y]<<"  ";
			out<<endl<<endl;
        }
	    out<<endl;
    }
    
	out.close();//data saved--close connection

	return;
}
/*
	--LoadSavedData--
	Load the network weights and bias values which were able to achieve optimal results from file
*/
void NeuralNetwork::LoadSavedData(Sizes Layersize,char* filename)//load saved data from file
{
 	ifstream in(filename);
    if(!in) {
		cout << endl << "failed to save file" << endl;//error reading from file
		return;
    }
    
	for(int layer=0; layer < Layersize.size()-1; layer++)//read weights
		for(int row  = 0; row <Layersize[layer] ; row ++)
			for(int col = 0; col < Layersize[layer+1]; col++)
				in>>nLayer[layer].Weights[row ][col];


	for( int layer=1; layer < Layersize.size(); layer++)//read bias
		for(int y = 0 ; y < Layersize[layer]; y++)
			in>>	nLayer[layer].Bias[y] ;

	in.close();
	cout << endl << "data loaded for testing" << endl;//data read...close connection 
	return;
 }
 

double NeuralNetwork::CountTestingData(TrainingExamples TraineeSamples,int temp,Sizes Layersize)
{
	//variable declaration
    double count = 0;
    int total = TraineeSamples.Datapoints;
    double accepted =  temp * 1;
    double desiredoutput;
    double actualoutput;
    double Error;
    int end = Layersize.size() - 1;

    for(int pattern = 0; pattern< temp; pattern++){
		//variable declaration
		Layer Desired;//to hold desired output from dataset
		Layer Actual;//to hold actual calculated output

		for(int i = 0; i <  Layersize[end] ;i++)
			Desired.push_back(0);//initiliaze with 0s
		for(int j = 0; j <  Layersize[end] ;j++)
			Actual.push_back(0);//intialize with 0s

		for(int output = 0; output < Layersize[end]; output++) {
			desiredoutput = TraineeSamples.OutputValues[pattern][output];
			actualoutput = Output[pattern][output];

			Desired[output] = desiredoutput;
			if((actualoutput >= 0)&&(actualoutput <= 0.5))
				actualoutput = 0;//if its between 0-0.5 then round it down to 0

			else if((actualoutput <= 1)&&(actualoutput >= 0.5))
				actualoutput = 1;//it its between 1 and 0.5 then round it up to 1

			Actual[output] =  actualoutput;//store new actual output value

		}
		
		int confirm = 0;
		
		for(int b = 0; b <  Layersize[end] ;b++){
			if(Desired[b]== Actual[b] )//check if the actual and desired output match i.e if the prediction/classification was correct
				confirm++;
        }
		
		if(confirm == Layersize[end])
			count++;//if an instance is correctly predicted meaning all the values of the output layer neurons match the desired output then increase count

		confirm = 0;//reset for next set

	}



  return count;//return count of correctly predicted instances

}


double NeuralNetwork::CountLearningData(TrainingExamples TraineeSamples,int temp,Sizes Layersize)
{
	//variable declaration
    double count = 0;
    int total = TraineeSamples.Datapoints;
    double accepted =  temp * 1;
    double desiredoutput;
    double actualoutput;
    double Error;
    int end = Layersize.size() - 1;

    for(int pattern = 0; pattern< temp; pattern++){

		Layer Desired;//to hold desired output from dataset
		Layer Actual;//to hold calculated output values

		for(int i = 0; i <  Layersize[end] ;i++)
			Desired.push_back(0);//initialize with 0s
		for(int j = 0; j <  Layersize[end] ;j++)
			Actual.push_back(0);//initialize with 0s



		for(int output = 0; output < Layersize[end]; output++) {
			desiredoutput = TraineeSamples.OutputValues[pattern][output];
			actualoutput = Output[pattern][output];

			Desired[output] = desiredoutput;

			
			if((actualoutput >= 0)&&(actualoutput <= (0+error_tolerance)))
				actualoutput = 0;

			else if((actualoutput <= 1)&&(actualoutput >= (1-error_tolerance)))
				actualoutput = 1;

			Actual[output] =  actualoutput;//set new actual output values

		}
     
		int confirm = 0;
		
		for(int b = 0; b <  Layersize[end] ;b++){
			if(Desired[b]== Actual[b] )
				confirm++;//match
        }
		
		if(confirm == Layersize[end])
			count++;//if an instance is correctly predicted meaning all the values of the output layer neurons match the desired output then increase count

		confirm = 0;

    }



	return count;

}
/*
	--CheckOutput--
	To see if actual and desired output values match
*/
bool NeuralNetwork::CheckOutput(TrainingExamples TraineeSamples,int pattern,Sizes Layersize)
{
	//variable declaration
    int end = Layersize.size() - 1;//know last layer
    double desiredoutput;
    double actualoutput;
	Layer Desired; //to hold desired output from dataset
    Layer Actual; //to hold actual calculated output

    for(int i = 0; i <  Layersize[end] ;i++)
        Desired.push_back(0);//initialize with 0s
    for(int j = 0; j <  Layersize[end] ;j++)
        Actual.push_back(0);//initialize with 0s

    int count = 0;
    
    for(int output = 0; output < Layersize[end]; output++) {
		desiredoutput = TraineeSamples.OutputValues[pattern][output];
		actualoutput = Output[pattern][output];
		Desired[output] = desiredoutput;
		cout<< "desired : "<<desiredoutput<<"      "<<actualoutput<<endl;

		
		if((actualoutput >= 0)&&(actualoutput <= 0.5))
			actualoutput = 0;

		else if((actualoutput <= 1)&&(actualoutput >= 0.5))
			actualoutput = 1;

		Actual[output] =  actualoutput;//new actual output value
    }

    cout<<"---------------------"<<endl;

    for(int b = 0; b <  Layersize[end] ;b++){
		if(Desired[b]!= Actual[b] )//if the actual and intended output do not match then return false
			return false;
    }
	return true;
}
/*
	--TestTrainingData--
	Test the trained network with testing data
*/       
double NeuralNetwork::TestTrainingData(Sizes Layersize, char* filename,int  size, char* load, int inputsize, int outputsize,ofstream & out2  )
{
    //variable declaration
	bool valid;
    double count = 1;
    int total;
    double accuracy;
	int end = Layersize.size() - 1;

	//load testing data
    TrainingExamples Test(load,size,inputsize+outputsize ,   inputsize,  outputsize ); 
	//initialize network
    CreateNetwork(Layersize,Test);
	//load saved training data
    LoadSavedData(Layersize,filename); 
	
    for(int pattern = 0;pattern < size ;pattern++){

		ForwardPass(Test,pattern,Layersize);
    }
	
	for(int pattern = 0; pattern< size; pattern++){
		for(int output = 0; output < Layersize[end]; output++) {

			out2<< Output[pattern][output]   <<" "<<Test.OutputValues[pattern][output]<<" "<<fabs( fabs(Test.OutputValues[pattern][output] )-fabs    (Output[pattern][output]))<<endl;
      
		}  
	}
	out2<<endl;
	out2<<endl; 
	//get accuracy of test run
	accuracy = SumSquaredError(Test,Layersize);  
	out2<<" RMSE:  " <<accuracy<<endl;
	cout<<"RMSE: " <<accuracy<<" %"<<endl;
	
	return accuracy;

}
/*
	--TestLearnedData--
	Test the network using the training data
*/
double NeuralNetwork::TestLearnedData(Sizes Layersize, char* filename,int  size, char* load, int inputsize, int outputsize )
{
	//variable declaration
    bool valid;
    double count = 1;
    double total;
    double accuracy;
	//get testing data set
    TrainingExamples Test(filename,size,inputsize+outputsize, inputsize, outputsize);
     
    total = Test.InputValues.size(); //how many samples to test?
	//initialize network
    CreateNetwork(Layersize,Test);
	//load saved network data
    LoadSavedData(Layersize,load);
	
    for(int pattern = 0;pattern < total ;pattern++){

		ForwardPass(Test,pattern,Layersize);
    }
    
	count = CountTestingData(Test,size,Layersize);//get number of correctly predicted instances

	accuracy = (count/total)* 100;//get accuracy percentage
	cout<<"The sucessful count is "<<count<<" out of "<<total<<endl;
	cout<<"The accuracy of test is: " <<accuracy<<" %"<<endl;
	//return accuracy %
	return accuracy;

}

int NeuralNetwork::BackPropogation(  TrainingExamples TraineeSamples, double LearningRate,Sizes Layersize, char * Savefile, bool load)
{
    //variable declaration
	double SumErrorSquared;
	int Id = 0;
	int Epoch = 0;
	bool Learn = true;


    CreateNetwork(Layersize,TraineeSamples );//structure the network and initialize the weights & neurons
    cout<< " xx " <<endl;


    while( Learn == true){//learning from training data

		for(int pattern = 0; pattern < TraineeSamples.InputValues.size(); pattern++)//keep doing for all training values
		{
			ForwardPass( TraineeSamples,pattern,Layersize);//pass through the network and check output

			BackwardPass(TraineeSamples,LearningRate,pattern,Layersize);//check back for error
																
		}


		Epoch++;//number of iterations
		cout<<Epoch<< " : is Epoch    *********************    "<<endl;//output iteration no.

		SumErrorSquared = SumSquaredError(TraineeSamples,Layersize);//calculate sum squared error
		cout<<SumErrorSquared<< " : is SumErrorSquared"<<endl;//show error--error should gradually decrease


   
		double count = CountLearningData(TraineeSamples,TraineeSamples.InputValues.size(), Layersize);//get count of correctly predicted instances
		SaveLearnedData(Layersize, Savefile);//save weights and network structure to file
		double trained= count/TraineeSamples.InputValues.size()*100; //get percentage of correctly predicted instances
		cout<<trained<<" is percentage trained"<<endl;//ouput training percentage

		if(trained>=MinTrain){//if accuracy is greater than or equal to 95% then stop learning
			Learn = false;
        }


		if(Epoch ==  MaxTime  ){  
			Learn = false;
		}

	}
	return Epoch;//return no. of iterations
}


int main(void)
{
 
         char  * trainfile = "train.txt";// training data set
         char * testfile= "test.txt"; //testing data set
         char * saveknowledge = "learntweights.txt";
         int trainsize = 138;
         int testsize = 41;

         int inputsize = 13;
         int hidden = 10;
         int output = 3;
         double learningrate = 0.1;


	TrainingExamples Samples(trainfile, trainsize,inputsize+output, inputsize,output);  //get data from file 138-lines 16-values_in_line 13-data_input_values 3_output_values  (wine UCI dataset)

	Samples.printData();
       NeuralNetwork network;
 

		
		Sizes NNtopology; //vector of network topology [input,hidden,output] - would work with more input layer [input, hidden, hidden, output]. Note "Sizes" is typdef of vector <int>
		//number of neurons in each layer
		NNtopology.push_back(inputsize);//input
		NNtopology.push_back(hidden);//hidden
		NNtopology.push_back(output);//output
   
		 
		network.BackPropogation(Samples,learningrate,NNtopology, saveknowledge,true); 
		network.TestLearnedData(NNtopology,testfile,testsize, saveknowledge,  inputsize, output ); 
		 
                  //system("PAUSE");
	return 0;   
};
