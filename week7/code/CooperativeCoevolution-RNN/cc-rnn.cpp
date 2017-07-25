
// Dr. Rohitash Chandra, 2009.  Cooperative coevolution of Elman recurrent neural networks for chaotic time series

//applied for time series prediction, could be adapted for pattern classification problems. See the format of the dataset. 

//built using RNN:https://github.com/rohitash-chandra/VanillaElmanRNN
//built on G3-PCX: https://github.com/rohitash-chandra/G3-PCX-Evolutionary-Alg
//buling using CC: https://github.com/rohitash-chandra/CooperativeCoevolution

//Publication of results: 

//Rohitash Chandra, Mengjie Zhang, Cooperative coevolution of Elman recurrent neural networks for chaotic time series prediction, Neurocomputing, Volume 86, 1 June 2012, Pages 116-123, ISSN 0925-2312, http://dx.doi.org/10.1016/j.neucom.2012.01.014.
//(www.sciencedirect.com/science/article/pii/S0925231212001014)  http://repository.usp.ac.fj/7551/1/ccrnntime.pdf






#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <ctime>
#include<stdio.h>
#include<ctype.h>

#define neuronlevel   //neuron level decomposition is used
#define sigmoid //tanh, sigmoid for output layer
#define rosen          // choose the function:
#define EPSILON 1e-30
#define MAXFUN 50000000  //upper bound for number of function evaluations
#define MINIMIZE 1      //set 1 to minimize and -1 to maximize
#define LIMIT 1e-20     //accuracy of best solution fitness desired
#define KIDS 2          //pool size of kids to be formed (use 2,3 or 4)
#define M 1             //M+2 is the number of parents participating in xover (use 1)
#define family 2        //number of parents to be replaced by good individuals(use 1 or 2)
#define sigma_zeta 0.1
#define sigma_eta 0.1   //variances used in PCX (best if fixed at these values)
#define  NoSpeciesCons 300
#define NPSize KIDS + 2   //new pop size
#define RandParent M+2     //number of parents participating in PCX
#define MAXRUN 10      //number of runs each with different random initial population

time_t TicTime;
time_t TocTime;
using namespace ::std;

typedef vector<double> Layer;
typedef vector<double> Nodes;
typedef vector<double> Frame;
typedef vector<int> Sizes;
typedef vector<vector<double> > Weight;
typedef vector<vector<double> > Data;

//Mackey Glass Data is used
const int trainsize = 299; //255
const int testsize = 99; //16000
char * trainfile = "train_embed.txt"; // may need to update this path to your own pc
char * testfile = "test_embed.txt"; //   may need to update this path to your own pc
const double MAEscalingfactor = 10;

int maxgen = 1000; //max Func eval (training time)
const int PopSize = 200;
const double MinimumError = 0.00001;
const int LayersNumber = 3; //total number of layers.
const int MaxVirtLayerSize = 20; //max number of unfold by RNN
const double MaxErrorTollerance = 0.20;
const double MomentumRate = 0;
const double Beta = 0;
double weightdecay = 0.005;

int row;
int col;
int layer;
int r;
int x;
int y;
double d_not[PopSize];
double seed, basic_seed;
int RUN;

class Samples {

public:
	Data InputValues;
	Data DataSet;
	Layer OutputValues;
	int PhoneSize;

public:
	Samples() {
	}
};

typedef vector<Samples> DataSample;

class TrainingExamples {

public:
	char* FileName;
	int SampleSize;
	int ColumnSize;
	int OutputSize;
	int RowSize;
	DataSample Sample;
public:
	TrainingExamples() {
	}
	;

	TrainingExamples(char* File, int sampleSize, int columnSize,
			int outputSize) {
		Samples sample;

		for (int i = 0; i < sampleSize; i++) {
			Sample.push_back(sample);
		}

		int rows;
		RowSize = MaxVirtLayerSize; // max number of rows.
		ColumnSize = columnSize;
		SampleSize = sampleSize;
		OutputSize = outputSize;

		ifstream in(File);

		//initialise input vectors
		for (int sample = 0; sample < SampleSize; sample++) {

			for (int r = 0; r < RowSize; r++)
				Sample[sample].InputValues.push_back(vector<double>());

			for (int row = 0; row < RowSize; row++) {
				for (int col = 0; col < ColumnSize; col++)
					Sample[sample].InputValues[row].push_back(0);
			}

			for (int out = 0; out < OutputSize; out++)
				Sample[sample].OutputValues.push_back(0);
		}
		//---------------------------------------------

		for (int samp = 0; samp < SampleSize; samp++) {
			in >> rows;
			Sample[samp].PhoneSize = rows;

			for (row = 0; row < Sample[samp].PhoneSize; row++) {
				for (col = 0; col < ColumnSize; col++)
					in >> Sample[samp].InputValues[row][col];
			}

			for (int out = 0; out < OutputSize; out++)
				in >> Sample[samp].OutputValues[out];

		}

		cout << "printing..." << endl;
		in.close();
	}
	void printData();
};
//.................................................

void TrainingExamples::printData() {
	for (int sample = 0; sample < SampleSize; sample++) {
		for (row = 0; row < Sample[sample].PhoneSize; row++) {
			for (col = 0; col < ColumnSize; col++)
				cout << Sample[sample].InputValues[row][col] << " ";
			cout << endl;
		}
		cout << endl;
		for (int out = 0; out < OutputSize; out++)
			cout << " " << Sample[sample].OutputValues[out] << " ";

		cout << endl << "--------------" << endl;
	}
}

//*********************************************************
class Layers {

public:
	double Weights[35][35];
	double WeightChange[35][35];
	double ContextWeight[35][35];
	Weight TransitionProb;
	Data RadialOutput;
	Data Outputlayer;
	Layer Bias;
	Layer BiasChange;
	Data Error;
	Layer Mean;
	Layer StanDev;
	Layer MeanChange;
	Layer StanDevChange;

public:
	Layers() {
	}

};

//***************************************************

class NeuralNetwork: public virtual TrainingExamples {

public:
	Layers nLayer[LayersNumber];
	double Heuristic;
	Layer ChromeNeuron;
	Data Output;
	double NMSE;
	int StringSize;
	Sizes layersize;

public:
	NeuralNetwork(Sizes layer) {
		layersize = layer;
		StringSize = (layer[0] * layer[1]) + (layer[1] * layer[2])
				+ (layer[1] * layer[1]) + (layer[1] + layer[2]);
	}
	NeuralNetwork() {
	}

	double Random();
	double Sigmoid(double ForwardOutput);
	double SigmoidS(double ForwardOutput);
	double NMSError() {
		return NMSE;
	}
	void CreateNetwork(Sizes Layersize, int Maxsize);
	void ForwardPass(Samples Sample, int patternNum, Sizes Layersize,
			int phone);
	void BackwardPass(Samples Sample, double LearningRate, int slide,
			Sizes Layersize, int phone);
	void PrintWeights(Sizes Layersize); // print  all weights
	bool ErrorTolerance(TrainingExamples TraineeSamples, int temp,
			Sizes Layersize);
	double SumSquaredError(TrainingExamples TraineeSamples, int temp,
			Sizes Layersize);
	int BackPropogation(TrainingExamples TraineeSamples, double LearningRate,
			Sizes Layersize, char* Savefile, char* TestFile, int sampleSize,
			int columnSize, int outputSize);
	Layer Neurons_to_chromes();
	void SaveLearnedData(Sizes Layersize, char* filename);
	void LoadSavedData(Sizes Layersize, char* filename);
	double TestLearnedData(Sizes Layersize, char* learntData, char* TestFile,
			int sampleSize, int columnSize, int outputSize);
	 
	double CountLearningData(TrainingExamples TraineeSamples, int temp,
			Sizes Layersize);
	double CountTestingData(TrainingExamples TraineeSamples, int temp,
			Sizes Layersize);
	 
	double MAE(TrainingExamples TraineeSamples, int temp, Sizes Layersize);
	void ChoromesToNeurons(Layer NeuronChrome);
	double ForwardFitnessPass(Layer NeuronChrome, TrainingExamples Test);
	bool CheckOutput(TrainingExamples TraineeSamples, int pattern,
			Sizes Layersize);
	 
	double TestTrainingData(Sizes Layersize, char* learntData, char* TestFile,
			int sampleSize, int columnSize, int outputSize, ofstream & out2);
	 
	double NormalisedMeanSquaredError(TrainingExamples TraineeSamples, int temp,
			Sizes Layersize);
	double BP(Layer NeuronChrome, TrainingExamples Test, int generations);
	double Abs(double num);
	 
	double MAPE(TrainingExamples TraineeSamples, int temp, Sizes Layersize);

};

double NeuralNetwork::Random() {
	int chance;
	double randomWeight = 0;
	double NegativeWeight = 0;
	chance = rand() % 2;

	if (chance == 0) {
		randomWeight = rand() % 100;
		return randomWeight * 0.05;
	}

	if (chance == 1) {
		NegativeWeight = rand() % 100;
		return NegativeWeight * 0.05;
	}

}
double NeuralNetwork::Sigmoid(double ForwardOutput) {
	double ActualOutput;
#ifdef sigmoid
	ActualOutput = (1.0 / (1.0 + exp(-1.0 * (ForwardOutput))));
#endif

#ifdef tanh
	ActualOutput = (exp(2 * ForwardOutput) - 1)/(exp(2 * ForwardOutput) + 1);
#endif
	return ActualOutput;
}

double NeuralNetwork::SigmoidS(double ForwardOutput) {
	double ActualOutput;

	ActualOutput = (1.0 / (1.0 + exp(-1.0 * (ForwardOutput))));

	return ActualOutput;
}

double NeuralNetwork::Abs(double num) {
	if (num < 0)
		return num * -1;
	else
		return num;
} 

 
void NeuralNetwork::CreateNetwork(Sizes Layersize, int Maxsize) {
	int end = Layersize.size() - 1;

	for (layer = 0; layer < Layersize.size() - 1; layer++) {

		for (row = 0; row < Layersize[layer]; row++)
			for (col = 0; col < Layersize[layer + 1]; col++)
				nLayer[layer].Weights[row][col] = Random();

		for (row = 0; row < Layersize[layer]; row++)
			for (col = 0; col < Layersize[layer + 1]; col++)
				nLayer[layer].WeightChange[row][col] = Random();
	}

	//------context layer weight initialisation-------------

	for (row = 0; row < Layersize[1]; row++)
		for (col = 0; col < Layersize[1]; col++)
			nLayer[1].ContextWeight[row][col] = Random();
	//------------------------------------------------------

	for (layer = 0; layer < Layersize.size(); layer++) {

		for (r = 0; r < Maxsize; r++)
			nLayer[layer].Outputlayer.push_back(vector<double>());

		for (row = 0; row < Maxsize; row++)
			for (col = 0; col < Layersize[layer]; col++)
				nLayer[layer].Outputlayer[row].push_back(Random());

		for (r = 0; r < MaxVirtLayerSize; r++)
			nLayer[layer].Error.push_back(vector<double>());

		for (row = 0; row < MaxVirtLayerSize; row++)
			for (col = 0; col < Layersize[layer]; col++)
				nLayer[layer].Error[row].push_back(0);

		//TransitionProb
		//---------------------------------------------
		for (r = 0; r < MaxVirtLayerSize; r++)
			nLayer[layer].RadialOutput.push_back(vector<double>());

		for (row = 0; row < MaxVirtLayerSize; row++)
			for (col = 0; col < Layersize[layer]; col++)
				nLayer[layer].RadialOutput[row].push_back(Random());

		//---------------------------------------------

		for (row = 0; row < Layersize[layer]; row++)
			nLayer[layer].Bias.push_back(Random());

		for (row = 0; row < Layersize[layer]; row++)
			nLayer[layer].BiasChange.push_back(0);

		for (row = 0; row < Layersize[layer]; row++)
			nLayer[layer].Mean.push_back(Random());

		for (row = 0; row < Layersize[layer]; row++)
			nLayer[layer].StanDev.push_back(Random());

		for (row = 0; row < Layersize[layer]; row++)
			nLayer[layer].MeanChange.push_back(0);

		for (row = 0; row < Layersize[layer]; row++)
			nLayer[layer].StanDevChange.push_back(0);

	}
	//--------------------------------------

	for (r = 0; r < Maxsize; r++)
		Output.push_back(vector<double>());
	for (row = 0; row < Maxsize; row++)
		for (col = 0; col < Layersize[end]; col++)
			Output[row].push_back(0);

	for (row = 0; row < StringSize; row++)
		ChromeNeuron.push_back(0);

}

void NeuralNetwork::ForwardPass(Samples Sample, int slide, Sizes Layersize,
		int phone) {
	double WeightedSum = 0;
	double ContextWeightSum = 0;
	double ForwardOutput;
	int end = Layersize.size() - 1;

	for (int row = 0; row < Layersize[0]; row++)
		nLayer[0].Outputlayer[slide + 1][row] = Sample.InputValues[slide][row];

	int layer = 0;
	int y;
	int x;
	for (y = 0; y < Layersize[layer + 1]; y++) {
		for (x = 0; x < Layersize[layer]; x++) {
			WeightedSum += (nLayer[layer].Outputlayer[slide + 1][x]
					* nLayer[layer].Weights[x][y]);
		}
		for (x = 0; x < Layersize[layer + 1]; x++) {
			ContextWeightSum += (nLayer[1].Outputlayer[slide][x]
					* nLayer[1].ContextWeight[x][y]); // adjust this line when use two hidden layers.
		}

		ForwardOutput = (WeightedSum + ContextWeightSum)
				- nLayer[layer + 1].Bias[y];
		nLayer[layer + 1].Outputlayer[slide + 1][y] = SigmoidS(ForwardOutput);

		WeightedSum = 0;
		ContextWeightSum = 0;
	}

	layer = 1;
	for (y = 0; y < Layersize[layer + 1]; y++) {
		for (x = 0; x < Layersize[layer]; x++) {
			WeightedSum += (nLayer[layer].Outputlayer[slide + 1][x]
					* nLayer[layer].Weights[x][y]);
			ForwardOutput = (WeightedSum) - nLayer[layer + 1].Bias[y];
		}
		nLayer[layer + 1].Outputlayer[slide + 1][y] = Sigmoid(ForwardOutput);
		WeightedSum = 0;
		ContextWeightSum = 0;
	}

	//--------------------------------------------
	for (int output = 0; output < Layersize[end]; output++) {
		Output[phone][output] = nLayer[end].Outputlayer[slide + 1][output];

	}

}

void NeuralNetwork::BackwardPass(Samples Sample, double LearningRate, int slide,
		Sizes Layersize, int phone) {

	int end = Layersize.size() - 1; // know the end layer
	double temp = 0;
	double sum = 0;
	int Endslide = Sample.PhoneSize;

	// compute error gradient for output neurons
	for (int output = 0; output < Layersize[end]; output++) {
		nLayer[2].Error[Endslide][output] = 1
				* (Sample.OutputValues[output] - Output[phone][output]);
	}

	int layer = 1;
	for (x = 0; x < Layersize[layer]; x++) { //inner layer
		for (y = 0; y < Layersize[layer + 1]; y++) { //outer layer
			temp += (nLayer[layer + 1].Error[Endslide][y]
					* nLayer[layer].Weights[x][y]);
		}

		nLayer[layer].Error[Endslide][x] =
				nLayer[layer].Outputlayer[Endslide][x]
						* (1 - nLayer[layer].Outputlayer[Endslide][x]) * temp;
		temp = 0.0;
	}

	for (x = 0; x < Layersize[1]; x++) { //inner layer
		for (y = 0; y < Layersize[1]; y++) { //outer layer
			sum += (nLayer[1].Error[slide][y] * nLayer[1].ContextWeight[x][y]);

		}
		nLayer[1].Error[slide - 1][x] = (nLayer[1].Outputlayer[slide - 1][x]
				* (1 - nLayer[1].Outputlayer[slide - 1][x])) * sum;
		sum = 0.0;
	}
	sum = 0.0;

// do weight updates..
//---------------------------------------
	double tmp;
	for (x = 0; x < Layersize[0]; x++) { //inner layer
		for (y = 0; y < Layersize[1]; y++) { //outer layer
			tmp = ((LearningRate * nLayer[1].Error[slide][y]
					* nLayer[0].Outputlayer[slide][x])); // weight change
			nLayer[0].Weights[x][y] += tmp - (tmp * weightdecay);
		}

	}

//-------------------------------------------------
//do top weight update
	double seeda = 0;

	seeda = 1;
	double tmpoo;
	for (x = 0; x < Layersize[1]; x++) { //inner layer
		for (y = 0; y < Layersize[2]; y++) { //outer layer
			tmpoo = ((seeda * LearningRate * nLayer[2].Error[Endslide][y]
					* nLayer[1].Outputlayer[Endslide][x])); // weight change
			nLayer[1].Weights[x][y] += tmpoo - (tmpoo * weightdecay);

		}

	}
	seeda = 0;

	//-----------------------------------------------
	double tmp2;
	for (x = 0; x < Layersize[1]; x++) { //inner layer
		for (y = 0; y < Layersize[1]; y++) { //outer layer
			tmp2 = ((LearningRate * nLayer[1].Error[slide][y]
					* nLayer[1].Outputlayer[slide - 1][x])); // weight change
			nLayer[1].ContextWeight[x][y] += tmp2 - (tmp2 * weightdecay);
		}
	}

//update the bias
	double topbias = 0;
	double seed = 0;

	seed = 1;
	for (y = 0; y < Layersize[2]; y++) {
		topbias = ((seed * -1 * LearningRate * nLayer[2].Error[Endslide][y]));
		nLayer[2].Bias[y] += topbias - (topbias * weightdecay);
		topbias = 0;
	}
	topbias = 0;
	seed = 0;

	double tmp1;
	for (y = 0; y < Layersize[1]; y++) {
		tmp1 = ((-1 * LearningRate * nLayer[1].Error[slide][y]));
		nLayer[1].Bias[y] += tmp1 - (tmp1 * weightdecay);
	}

}
double NeuralNetwork::CountLearningData(TrainingExamples TraineeSamples,
		int temp, Sizes Layersize) {
	double count = 1;
	int total = TraineeSamples.SampleSize;
	double accepted = temp * 1;

	double desiredoutput;
	double actualoutput;

	double Error;
	int end = Layersize.size() - 1;

	for (int pattern = 0; pattern < temp; pattern++) {

		Layer Desired;
		Layer Actual;

		for (int i = 0; i < Layersize[end]; i++)
			Desired.push_back(0);
		for (int j = 0; j < Layersize[end]; j++)
			Actual.push_back(0);

		for (int output = 0; output < Layersize[end]; output++) {
			desiredoutput = TraineeSamples.Sample[pattern].OutputValues[output];
			actualoutput = Output[pattern][output];

			Desired[output] = desiredoutput;

			if ((actualoutput >= 0) && (actualoutput <= 0.2))
				actualoutput = 0;

			else if ((actualoutput <= 1) && (actualoutput >= 0.8))
				actualoutput = 1;

			Actual[output] = actualoutput;

		}
		int confirm = 0;
		for (int b = 0; b < Layersize[end]; b++) {
			if (Desired[b] == Actual[b])
				confirm++;
		}
		if (confirm == Layersize[end])
			count++;

		confirm = 0;

	}

	return count;

}
double NeuralNetwork::CountTestingData(TrainingExamples TraineeSamples,
		int temp, Sizes Layersize) {
	double count = 1;
	int total = TraineeSamples.SampleSize;
	double accepted = temp * 1;

	double desiredoutput;
	double actualoutput;

	double Error;
	int end = Layersize.size() - 1;

	for (int pattern = 0; pattern < temp; pattern++) {

		Layer Desired;
		Layer Actual;

		for (int i = 0; i < Layersize[end]; i++)
			Desired.push_back(0);
		for (int j = 0; j < Layersize[end]; j++)
			Actual.push_back(0);

		for (int output = 0; output < Layersize[end]; output++) {
			desiredoutput = TraineeSamples.Sample[pattern].OutputValues[output];
			actualoutput = Output[pattern][output];

			Desired[output] = desiredoutput;
			if ((actualoutput >= 0) && (actualoutput <= 0.5))
				actualoutput = 0;

			else if ((actualoutput <= 1) && (actualoutput >= 0.5))
				actualoutput = 1;

			Actual[output] = actualoutput;

		}
		int confirm = 0;
		for (int b = 0; b < Layersize[end]; b++) {
			if (Desired[b] == Actual[b])
				confirm++;
		}
		if (confirm == Layersize[end])
			count++;

		confirm = 0;

	}
	return count;

}

bool NeuralNetwork::ErrorTolerance(TrainingExamples TraineeSamples, int temp,
		Sizes Layersize) {
	double count = 0;
	int total = TraineeSamples.SampleSize;
	double accepted = temp * 1;

	double desiredoutput;
	double actualoutput;

	double Error;
	int end = Layersize.size() - 1;

	for (int pattern = 0; pattern < temp; pattern++) {

		Layer Desired;
		Layer Actual;

		for (int i = 0; i < Layersize[end]; i++)
			Desired.push_back(0);
		for (int j = 0; j < Layersize[end]; j++)
			Actual.push_back(0);

		for (int output = 0; output < Layersize[end]; output++) {
			desiredoutput = TraineeSamples.Sample[pattern].OutputValues[output];
			actualoutput = Output[pattern][output];

			Desired[output] = desiredoutput;

			if ((actualoutput >= 0) && (actualoutput <= 0.2))
				actualoutput = 0;

			else if ((actualoutput <= 1) && (actualoutput >= 0.8))
				actualoutput = 1;

			Actual[output] = actualoutput;

		}
		int confirm = 0;
		for (int b = 0; b < Layersize[end]; b++) {
			if (Desired[b] == Actual[b])
				confirm++;
		}
		if (confirm == Layersize[end])
			count++;
		confirm = 0;
	}
	if (count == accepted)
		return false;
	return true;

}

double NeuralNetwork::MAPE(TrainingExamples TraineeSamples, int temp,
		Sizes Layersize) {
	int end = Layersize.size() - 1;
	double Sum = 0;
	double Error = 0;
	double ErrorSquared = 0;
	for (int pattern = 0; pattern < temp; pattern++) {
		for (int output = 0; output < Layersize[end]; output++) {
			Error = (TraineeSamples.Sample[pattern].OutputValues[output]
					- Output[pattern][output])
					/ TraineeSamples.Sample[pattern].OutputValues[output];
			ErrorSquared += fabs(Error);
		}
		Sum += (ErrorSquared);
		ErrorSquared = 0;
	}
	return (Sum / temp * Layersize[end] * 100);
}

double NeuralNetwork::MAE(TrainingExamples TraineeSamples, int temp,
		Sizes Layersize) {
	int end = Layersize.size() - 1;
	double Sum = 0;
	double Error = 0;
	double ErrorSquared = 0;
	for (int pattern = 0; pattern < temp; pattern++) {
		for (int output = 0; output < Layersize[end]; output++) {
			Error = (TraineeSamples.Sample[pattern].OutputValues[output]
					- Output[pattern][output]) * MAEscalingfactor;

			ErrorSquared += fabs(Error);
		}

		Sum += (ErrorSquared);
		ErrorSquared = 0;
	}
	return Sum / temp * Layersize[end];

}
double NeuralNetwork::SumSquaredError(TrainingExamples TraineeSamples, int temp,
		Sizes Layersize) {
	int end = Layersize.size() - 1;
	double Sum = 0;
	double Error = 0;
	double ErrorSquared = 0;
	for (int pattern = 0; pattern < temp; pattern++) {
		for (int output = 0; output < Layersize[end]; output++) {
			Error = TraineeSamples.Sample[pattern].OutputValues[output]
					- Output[pattern][output];

			ErrorSquared += (Error * Error);
		}

		Sum += (ErrorSquared);
		ErrorSquared = 0;
	}
	return sqrt(Sum / temp * Layersize[end]);

}

double NeuralNetwork::NormalisedMeanSquaredError(
		TrainingExamples TraineeSamples, int temp, Sizes Layersize) {
	int end = Layersize.size() - 1;
	double Sum = 0;
	double Sum2 = 0;
	double Error = 0;
	double ErrorSquared = 0;
	double Error2 = 0;
	double ErrorSquared2 = 0;
	double meany = 0;
	for (int pattern = 0; pattern < temp; pattern++) {

		for (int slide = 0; slide < TraineeSamples.Sample[pattern].PhoneSize;
				slide++) {
			for (int input = 0; input < Layersize[0]; input++) {
				meany +=
						TraineeSamples.Sample[pattern].InputValues[slide][input];
			}
			meany /= Layersize[0] * TraineeSamples.Sample[pattern].PhoneSize;
		}

		for (int output = 0; output < Layersize[end]; output++) {
			Error2 = TraineeSamples.Sample[pattern].OutputValues[output]
					- meany;
			Error = TraineeSamples.Sample[pattern].OutputValues[output]
					- Output[pattern][output];
			ErrorSquared += (Error * Error);
			ErrorSquared2 += (Error2 * Error2);

		}
		meany = 0;
		Sum += (ErrorSquared);
		Sum2 += (ErrorSquared2);
		ErrorSquared = 0;
		ErrorSquared2 = 0;
	}

	return Sum / Sum2;
}

void NeuralNetwork::PrintWeights(Sizes Layersize) {
	int end = Layersize.size() - 1;

	for (int layer = 0; layer < Layersize.size() - 1; layer++) {

		cout << layer << "  Weights::" << endl << endl;
		for (int row = 0; row < Layersize[layer]; row++) {
			for (int col = 0; col < Layersize[layer + 1]; col++)
				cout << nLayer[layer].Weights[row][col] << " ";
			cout << endl;
		}
		cout << endl << layer << " ContextWeight::" << endl << endl;

		for (int row = 0; row < Layersize[1]; row++) {
			for (int col = 0; col < Layersize[1]; col++)
				cout << nLayer[1].ContextWeight[row][col] << " ";
			cout << endl;
		}

	}

}
//-------------------------------------------------------

void NeuralNetwork::SaveLearnedData(Sizes Layersize, char* filename) {

	ofstream out;
	out.open(filename);
	if (!out) {
		cout << endl << "failed to save file" << endl;
		return;
	}
	//-------------------------------
	for (int layer = 0; layer < Layersize.size() - 1; layer++) {
		for (int row = 0; row < Layersize[layer]; row++) {
			for (int col = 0; col < Layersize[layer + 1]; col++)
				out << nLayer[layer].Weights[row][col] << " ";
			out << endl;
		}
		out << endl;
	}
	//-------------------------------
	for (int row = 0; row < Layersize[1]; row++) {
		for (int col = 0; col < Layersize[1]; col++)
			out << nLayer[1].ContextWeight[row][col] << " ";
		out << endl;
	}
	out << endl;
	//--------------------------------
	// output bias.
	for (int layer = 1; layer < Layersize.size(); layer++) {
		for (int y = 0; y < Layersize[layer]; y++) {
			out << nLayer[layer].Bias[y] << "  ";
			out << endl;
		}

	}
	out << endl;
	out.close();
	return;
}

void NeuralNetwork::LoadSavedData(Sizes Layersize, char* filename) {
	ifstream in(filename);
	if (!in) {
		cout << endl << "failed to load file" << endl;
		return;
	}
	//-------------------------------
	for (int layer = 0; layer < Layersize.size() - 1; layer++)
		for (int row = 0; row < Layersize[layer]; row++)
			for (int col = 0; col < Layersize[layer + 1]; col++)
				in >> nLayer[layer].Weights[row][col];
	//---------------------------------
	for (int row = 0; row < Layersize[1]; row++)
		for (int col = 0; col < Layersize[1]; col++)
			in >> nLayer[1].ContextWeight[row][col];
	//--------------------------------
	// output bias.
	for (int layer = 1; layer < Layersize.size(); layer++)
		for (int y = 0; y < Layersize[layer]; y++)
			in >> nLayer[layer].Bias[y];

	in.close();
	return;
}

bool NeuralNetwork::CheckOutput(TrainingExamples TraineeSamples, int pattern,
		Sizes Layersize) {
	int end = Layersize.size() - 1;
	double desiredoutput;
	double actualoutput;

	Layer Desired;
	Layer Actual;

	for (int i = 0; i < Layersize[end]; i++)
		Desired.push_back(0);
	for (int j = 0; j < Layersize[end]; j++)
		Actual.push_back(0);

	//Layersize[end]
	for (int output = 0; output < Layersize[end]; output++) {
		desiredoutput = TraineeSamples.Sample[pattern].OutputValues[output];
		actualoutput = Output[pattern][output];

		Desired[output] = desiredoutput;

		cout << "desired : " << desiredoutput << "      " << actualoutput
				<< endl;

		if ((actualoutput >= 0) && (actualoutput <= 0.5))
			actualoutput = 0;

		else if ((actualoutput <= 1) && (actualoutput >= 0.5))
			actualoutput = 1;

		Actual[output] = actualoutput;
	}
	cout << "---------------------" << endl;
	for (int b = 0; b < Layersize[end]; b++) {
		if (Desired[b] != Actual[b])
			return false;
	}
	return true;
}

double NeuralNetwork::TestLearnedData(Sizes Layersize, char* learntData,
		char* TestFile, int sampleSize, int columnSize, int outputSize) {
	bool valid;
	double count = 1;
	double total;
	double accuracy;

	Samples sample;

	TrainingExamples Test(TestFile, sampleSize, columnSize, outputSize);

	total = Test.SampleSize; //how many samples to test?

	CreateNetwork(Layersize, sampleSize);

	LoadSavedData(Layersize, "Learnt.txt");

	for (int phone = 0; phone < Test.SampleSize; phone++) {
		sample = Test.Sample[phone];

		int slide;

		for (slide = 0; slide < sample.PhoneSize; slide++) {
			ForwardPass(sample, slide, Layersize, phone);

		}

	}
	count = CountTestingData(Test, sampleSize, Layersize);

	accuracy = (count / total) * 100;
	cout << "The sucessful count is " << count << " out of " << total << endl;
	cout << "The accuracy of test is: " << accuracy << " %" << endl;
	return accuracy;

}
 
double NeuralNetwork::TestTrainingData(Sizes Layersize, char* learntData,
		char* TestFile, int sampleSize, int columnSize, int outputSize,
		ofstream & out2) {
	bool valid;
	double count = 1;
	double total;
	double accuracy;
	int end = Layersize.size() - 1;
	Samples sample;

	TrainingExamples Test(TestFile, sampleSize, columnSize, outputSize);

	total = Test.SampleSize; //how many samples to test?

	CreateNetwork(Layersize, sampleSize);

	LoadSavedData(Layersize, "Learnt.txt");

	for (int phone = 0; phone < Test.SampleSize; phone++) {
		sample = Test.Sample[phone];

		int slide;

		for (slide = 0; slide < sample.PhoneSize; slide++) {
			ForwardPass(sample, slide, Layersize, phone);

		}
	}

	for (int pattern = 0; pattern < Test.SampleSize; pattern++) {
		out2 << Output[pattern][0] * 90 << " "
				<< Test.Sample[pattern].OutputValues[0] * 90 << " "
				<< (Test.Sample[pattern].OutputValues[0] - Output[pattern][0])
						* 90 << "     ";

		out2 << Output[pattern][1] * 360 << " "
				<< Test.Sample[pattern].OutputValues[1] * 360 << " "
				<< (Test.Sample[pattern].OutputValues[1] - Output[pattern][1])
						* 360 << endl;

	}
	out2 << endl;

	out2 << endl;
	accuracy = SumSquaredError(Test, Test.SampleSize, layersize);
	out2 << " RMSE:  " << accuracy << endl;
	cout << "RMSE: " << accuracy << " %" << endl;
	NMSE = MAE(Test, Test.SampleSize, layersize);
	out2 << " NMSE:  " << NMSE << endl;
	return accuracy;
}

int NeuralNetwork::BackPropogation(TrainingExamples TraineeSamples,
		double LearningRate, Sizes Layersize, char * Savefile, char* TestFile,
		int sampleSize, int columnSize, int outputSize) {
	ofstream out;
	out.open("lambgod.txt");
	ofstream outt;
	outt.open("BP.txt");
	double SumErrorSquared;
	Sizes Array;

	for (int i = 0; i < 300; i++)
		Array.push_back(10);
	Array[0] = 0;

	Samples sample;
	CreateNetwork(Layersize, TraineeSamples.SampleSize);
	int Id = 0;
	int final = 200;
	double Test = 0;
	double Testall = 0;
	bool Learn = true;
	int c = 1;
	int temp = 30;
	int cycle = 0;
	for (cycle = 0; cycle < 300; cycle++) {

		cout << endl << endl << "    " << cycle
				<< "----------------------------" << endl;
		temp += Array[cycle];
		for (int epoch = 0; epoch < final; epoch++) {
			for (int phone = 0; phone < temp; phone++) {
				sample = TraineeSamples.Sample[phone];

				nLayer[1].Outputlayer[0][0] = 0.5;
				nLayer[1].Outputlayer[0][1] = -0.5;

				int slide;

				for (slide = 0; slide < sample.PhoneSize; slide++) {

					ForwardPass(sample, slide, Layersize, phone);

				}

				for (slide = sample.PhoneSize; slide >= 1; slide--) {
					BackwardPass(sample, LearningRate, slide, Layersize, phone);
				}

			} //phone

			SumErrorSquared = SumSquaredError(TraineeSamples, temp, Layersize);

			cout << epoch << " : is Epoch    *********************    " << endl;

			cout << SumErrorSquared << " : is SumErrorSquared" << endl;

			Learn = ErrorTolerance(TraineeSamples, temp, Layersize);

			if (Learn == false)
				break;

		} //for epoch
		outt << SumErrorSquared << endl;
		SaveLearnedData(Layersize, Savefile);
		cout << " ---done after test train----" << endl;
		cout << " ---done after test all----" << endl;

		out << Test << "is tested all" << endl;
		out << temp << " was temp" << endl << endl;
		if (Test >= 100)
			break;
		if (final == 1000)
			break;
		if (temp >= (sampleSize - 20)) {
			final = 1000;

		}
		c++;
	} //for cycle
	cout << " ---done----" << endl;

	SaveLearnedData(Layersize, Savefile);

	out.close();

	outt.close();
	return c;

}
Layer NeuralNetwork::Neurons_to_chromes() {
	int gene = 0;
	Layer NeuronChrome(StringSize);
	int layer = 0;
	for (row = 0; row < layersize[layer]; row++) {
		for (col = 0; col < layersize[layer + 1]; col++) {
			NeuronChrome[gene] = nLayer[layer].Weights[row][col];
			gene++;
		}
	}

	layer = 1;
	for (row = 0; row < layersize[layer]; row++) {
		NeuronChrome[gene] = nLayer[layer].Bias[row];
		gene++;
	}

	for (row = 0; row < layersize[1]; row++) {
		for (col = 0; col < layersize[1]; col++) {
			NeuronChrome[gene] = nLayer[1].ContextWeight[row][col];
			gene++;
		}
	}
	layer = 1;
	for (row = 0; row < layersize[layer]; row++) {
		for (col = 0; col < layersize[layer + 1]; col++) {
			NeuronChrome[gene] = nLayer[layer].Weights[row][col];
			gene++;
		}
	}

	layer = 2;
	for (row = 0; row < layersize[layer]; row++) {
		NeuronChrome[gene] = nLayer[layer].Bias[row];
		gene++;
	}

	return NeuronChrome;
}

void NeuralNetwork::ChoromesToNeurons(Layer NeuronChrome) {
#ifdef neuronlevel
	int layer = 0;
	int gene = 0;

	for (int neu = 0; neu < layersize[1]; neu++) {

		for (int row = 0; row < layersize[layer]; row++) {
			nLayer[layer].Weights[row][neu] = NeuronChrome[gene];
			gene++;
		}

		nLayer[layer + 1].Bias[neu] = NeuronChrome[gene];
		gene++;
	}

	for (int neu = 0; neu < layersize[1]; neu++) {
		for (col = 0; col < layersize[1]; col++) {
			nLayer[1].ContextWeight[neu][col] = NeuronChrome[gene];
			gene++;
		}
	}

	for (int neu = 0; neu < layersize[2]; neu++) {

		for (int row = 0; row < layersize[layer + 1]; row++) {
			nLayer[layer + 1].Weights[row][neu] = NeuronChrome[gene];
			gene++;
		}

		nLayer[layer + 2].Bias[neu] = NeuronChrome[gene];
		gene++;

	}

#endif

}
double NeuralNetwork::ForwardFitnessPass(Layer NeuronChrome,
		TrainingExamples Test)

		{

	Samples sample;
	ChoromesToNeurons(NeuronChrome);

	double SumErrorSquared = 0;
	bool Learn = true;

	for (int phone = 0; phone < Test.SampleSize; phone++) {
		sample = Test.Sample[phone];
		int slide;

		for (slide = 0; slide < sample.PhoneSize; slide++) {
			ForwardPass(sample, slide, layersize, phone);
		}

	}

	SumErrorSquared = SumSquaredError(Test, Test.SampleSize, layersize);

	return SumErrorSquared;

}

double NeuralNetwork::BP(Layer NeuronChrome, TrainingExamples Test,
		int generations)

		{

	ChoromesToNeurons(NeuronChrome);
	Samples sample;
	double SumErrorSquared = 0;
	bool Learn = true;

	for (int epoch = 0; epoch < generations; epoch++) {

		for (int phone = 0; phone < Test.SampleSize; phone++) {
			sample = Test.Sample[phone];
			int slide;

			for (slide = 0; slide < sample.PhoneSize; slide++) {
				ForwardPass(sample, slide, layersize, phone);
			}

			for (slide = sample.PhoneSize; slide >= 1; slide--) {
				BackwardPass(sample, 1, slide, layersize, phone);
			}
		}

		SumErrorSquared = SumSquaredError(Test, Test.SampleSize, layersize);
		Learn = ErrorTolerance(Test, Test.SampleSize, layersize);
		if (Learn == false)
			return -1;
	}
	ChromeNeuron = Neurons_to_chromes();
	return SumErrorSquared;

}

//-------------------------------------------------------

class Individual {

public:

	Layer Chrome;
	double Fitness;
	Layer BitChrome;

public:
	Individual() {

	}
	void print();

};
class GeneticAlgorithmn {

public:
	Individual Population[PopSize];
	int TempIndex[PopSize];

	Individual NewPop[NPSize];
	int mom[PopSize];
	int list[NPSize];

	int MaxGen;

	int NumVariable;

	double BestFit;
	int BestIndex;
	int NumEval;

	int kids;

public:
	GeneticAlgorithmn(int stringSize) {
		NumVariable = stringSize;
		NumEval = 0;
		BestIndex = 0;
	}
	GeneticAlgorithmn() {
		BestIndex = 0;
	}

	double Fitness() {
		return BestFit;
	}

	double RandomWeights();

	double RandomAddition();

	void PrintPopulation();

	int GenerateNewPCX(int pass, NeuralNetwork network, TrainingExamples Sample,
			double Mutation, int depth);

	double Objective(Layer x);

	void InitilisePopulation();

	void Evaluate();

	double modu(double index[]);

	double innerprod(double Ind1[], double Ind2[]);

	double RandomParents();

	double MainAlgorithm(double RUN, ofstream &out1, ofstream &out2,
			ofstream &out3);

	double Noise();
	double rand_normal(double mean, double stddev);

	void my_family(); //here a random family (1 or 2) of parents is created who would be replaced by good individuals

	void find_parents();
	void rep_parents(); //here the best (1 or 2) individuals replace the family of parents
	void sort();

};

double GeneticAlgorithmn::RandomWeights() {
	int chance;
	double randomWeight;
	double NegativeWeight;
	chance = rand() % 2;

	if (chance == 0) {
		randomWeight = rand() % 100000;
		return randomWeight * 0.00005;
	}

	if (chance == 1) {
		NegativeWeight = rand() % 100000;
		return NegativeWeight * -0.00005;
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
			x = 2.0 * rand() / RAND_MAX - 1;
			y = 2.0 * rand() / RAND_MAX - 1;

			r = x * x + y * y;
		} while (r == 0.0 || r > 1.0);

		{
			// Apply Box-Muller transform on x, y
			double d = sqrt(-2.0 * log(r) / r);
			double n1 = x * d;
			n2 = y * d;

			// scale and translate to get desired mean and standard deviation
			double result = n1 * stddev + mean;

			n2_cached = 1;
			return result;
		}
	} else {
		n2_cached = 0;
		return n2 * stddev + mean;
	}
}

double GeneticAlgorithmn::RandomAddition() {
	int chance;
	double randomWeight;
	double NegativeWeight;
	chance = rand() % 2;

	if (chance == 0) {
		randomWeight = rand() % 100;
		return randomWeight * 0.009;
	}

	if (chance == 1) {
		NegativeWeight = rand() % 100;
		return NegativeWeight * -0.009;
	}

}

void GeneticAlgorithmn::InitilisePopulation() {

	double x, y;

	for (int row = 0; row < PopSize; row++)
		TempIndex[row] = 0;

	for (int row = 0; row < PopSize; row++) {
		for (int col = 0; col < NumVariable; col++) {

			Population[row].Chrome.push_back(RandomWeights());
		}
	}
	for (int row = 0; row < NPSize; row++) {
		for (int col = 0; col < NumVariable; col++)
			NewPop[row].Chrome.push_back(0);

	}
}

void GeneticAlgorithmn::Evaluate() {
	Population[0].Fitness = Objective(Population[0].Chrome);
	BestFit = Population[0].Fitness;
	BestIndex = 0;

	for (int row = 0; row < PopSize; row++) {
		Population[row].Fitness = Objective(Population[row].Chrome);
		if ((MINIMIZE * BestFit) > (MINIMIZE * Population[row].Fitness)) {
			BestFit = Population[row].Fitness;
			BestIndex = row;
		}
	}
}

void GeneticAlgorithmn::PrintPopulation() {
	for (int row = 0; row < PopSize; row++) {
		for (int col = 0; col < NumVariable; col++)
			cout << Population[row].Chrome[col] << " ";
		cout << endl;
	}

	for (int row = 0; row < PopSize; row++)
		cout << Population[row].Fitness << endl;

	cout << " ---" << endl;
	cout << BestFit << "  " << BestIndex << endl;

}

double GeneticAlgorithmn::Objective(Layer x) {
	int i, j, k;
	double fit, sumSCH;

	fit = 0.0;

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
	//Rosenbrock's function
	for (j = 0; j < NumVariable - 1; j++)
		fit += 100.0 * (x[j] * x[j] - x[j + 1]) * (x[j] * x[j] - x[j + 1])
				+ (x[j] - 1.0) * (x[j] - 1.0);

	NumEval++;
#endif

	// #ifdef 6legkine

	return (fit);
}
//------------------------------------------------------------------------

void GeneticAlgorithmn::my_family() //here a random family (1 or 2) of parents is created who would be replaced by good individuals
{
	int i, j, index;
	int swp;
	double u;

	for (i = 0; i < PopSize; i++)
		mom[i] = i;

	for (i = 0; i < family; i++) {

		index = (rand() % PopSize) + i;
		if (index > (PopSize - 1))
			index = PopSize - 1;
		swp = mom[index];
		mom[index] = mom[i];
		mom[i] = swp;
	}
}

void GeneticAlgorithmn::find_parents() //here the parents to be replaced are added to the temporary sub-population to assess their goodness against the new solutions formed which will be the basis of whether they should be kept or not
{
	int i, j, k;
	double u, v;

	my_family();
//cout<<kids<<endl;
	for (j = 0; j < family; j++) {
		for (i = 0; i < NumVariable; i++)
			NewPop[kids + j].Chrome[i] = Population[mom[j]].Chrome[i];

		NewPop[kids + j].Fitness = Objective(NewPop[kids + j].Chrome);

	}
}

void GeneticAlgorithmn::rep_parents() //here the best (1 or 2) individuals replace the family of parents
{
	int i, j;
	for (j = 0; j < family; j++) {
		for (i = 0; i < NumVariable; i++)
			Population[mom[j]].Chrome[i] = NewPop[list[j]].Chrome[i];

		Population[mom[j]].Fitness = Objective(Population[mom[j]].Chrome);

	}
}

void GeneticAlgorithmn::sort()

{
	int i, j, temp;
	double dbest;

	for (i = 0; i < (kids + family); i++)
		list[i] = i;

	if (MINIMIZE)
		for (i = 0; i < (kids + family - 1); i++) {
			dbest = NewPop[list[i]].Fitness;
			for (j = i + 1; j < (kids + family); j++) {
				if (NewPop[list[j]].Fitness < dbest) {
					dbest = NewPop[list[j]].Fitness;
					temp = list[j];
					list[j] = list[i];
					list[i] = temp;
				}
			}
		}
	else
		for (i = 0; i < (kids + family - 1); i++) {
			dbest = NewPop[list[i]].Fitness;
			for (j = i + 1; j < (kids + family); j++) {
				if (NewPop[list[j]].Fitness > dbest) {
					dbest = NewPop[list[j]].Fitness;
					temp = list[j];
					list[j] = list[i];
					list[i] = temp;
				}
			}
		}
}

//---------------------------------------------------------------------
double GeneticAlgorithmn::modu(double index[]) {
	int i;
	double sum, modul;

	sum = 0.0;
	for (i = 0; i < NumVariable; i++)
		sum += (index[i] * index[i]);

	modul = sqrt(sum);
	return modul;
}

// calculates the inner product of two vectors
double GeneticAlgorithmn::innerprod(double Ind1[], double Ind2[]) {
	int i;
	double sum;

	sum = 0.0;

	for (i = 0; i < NumVariable; i++)
		sum += (Ind1[i] * Ind2[i]);

	return sum;
}

int GeneticAlgorithmn::GenerateNewPCX(int pass, NeuralNetwork network,
		TrainingExamples Sample, double Mutation, int depth) {
	int i, j, num, k;
	double Centroid[NumVariable];
	double tempvar, tempsum, D_not, dist;
	double tempar1[NumVariable];
	double tempar2[NumVariable];
	double D[RandParent];
	double d[NumVariable];
	double diff[RandParent][NumVariable];
	double temp1, temp2, temp3;
	int temp;

	for (i = 0; i < NumVariable; i++)
		Centroid[i] = 0.0;

	// centroid is calculated here
	for (i = 0; i < NumVariable; i++) {
		for (j = 0; j < RandParent; j++)
			Centroid[i] += Population[TempIndex[j]].Chrome[i];

		Centroid[i] /= RandParent;

	}
	for (j = 1; j < RandParent; j++) {
		for (i = 0; i < NumVariable; i++) {
			if (j == 1)
				d[i] = Centroid[i] - Population[TempIndex[0]].Chrome[i];
			diff[j][i] = Population[TempIndex[j]].Chrome[i]
					- Population[TempIndex[0]].Chrome[i];
		}
		if (modu(diff[j]) < EPSILON) {
			cout
					<< "RUN Points are very close to each other. Quitting this run   "
					<< endl;

			return (0);
		}

		if (isnan(diff[j][i])) {
			cout << "`diff nan   " << endl;
			diff[j][i] = 1;
			return (0);
		}

	}
	dist = modu(d); // modu calculates the magnitude of the vector

	if (dist < EPSILON) {
		cout << "RUN Points are very close to each other. Quitting this run    "
				<< endl;

		return (0);
	}

	// orthogonal directions are computed (see the paper)
	for (i = 1; i < RandParent; i++) {
		temp1 = innerprod(diff[i], d);
		if ((modu(diff[i]) * dist) == 0) {
			cout << " division by zero: part 1" << endl;
			temp2 = temp1 / (1);
		} else {
			temp2 = temp1 / (modu(diff[i]) * dist);
		}

		temp3 = 1.0 - pow(temp2, 2.0);
		D[i] = modu(diff[i]) * sqrt(temp3);
	}

	D_not = 0;
	for (i = 1; i < RandParent; i++)
		D_not += D[i];

	D_not /= (RandParent - 1); //this is the average of the perpendicular distances from all other parents (minus the index parent) to the index vector

	// Next few steps compute the child, by starting with a random vector
	for (j = 0; j < NumVariable; j++) {
		//tempar1[j]=noise(0.0,(D_not*sigma_eta));
		tempar1[j] = rand_normal(0, D_not * sigma_eta);
		//tempar1[j] = Noise();
		tempar2[j] = tempar1[j];
	}

	for (j = 0; j < NumVariable; j++) {
		if (pow(dist, 2.0) == 0) {
			cout << " division by zero: part 2" << endl;
			tempar2[j] = tempar1[j] - ((innerprod(tempar1, d) * d[j]) / 1);
		} else
			tempar2[j] = tempar1[j]
					- ((innerprod(tempar1, d) * d[j]) / pow(dist, 2.0));
	}

	for (j = 0; j < NumVariable; j++)
		tempar1[j] = tempar2[j];

	for (k = 0; k < NumVariable; k++)
		NewPop[pass].Chrome[k] = Population[TempIndex[0]].Chrome[k]
				+ tempar1[k];

	tempvar = rand_normal(0, sigma_zeta);

	for (k = 0; k < NumVariable; k++) {
		NewPop[pass].Chrome[k] += (tempvar * d[k]);

	}

	double random = rand() % 10;

	Layer Chrome(NumVariable);

	for (k = 0; k < NumVariable; k++) {
		if (!isnan(NewPop[pass].Chrome[k])) {
			Chrome[k] = NewPop[pass].Chrome[k];
		} else
			NewPop[pass].Chrome[k] = RandomAddition();

	}

	return (1);
}

//------------------------------------------------------------------------
double GeneticAlgorithmn::RandomParents() {

	int i, j, index;
	int swp;
	double u;
	int delta;

	for (i = 0; i < PopSize; i++)
		TempIndex[i] = i;

	swp = TempIndex[0];
	TempIndex[0] = TempIndex[BestIndex]; // best is always included as a parent and is the index parent
	// this can be changed for solving a generic problem
	TempIndex[BestIndex] = swp;

	for (i = 1; i < RandParent; i++) // shuffle the other parents
			{
		index = (rand() % PopSize) + i;

		if (index > (PopSize - 1))
			index = PopSize - 1;
		swp = TempIndex[index];
		TempIndex[index] = TempIndex[i];
		TempIndex[i] = swp;
	}

}

void TIC(void) {
	TicTime = time(NULL);
}

void TOC(void) {
	TocTime = time(NULL);
}

double StopwatchTimeInSeconds() {
	return difftime(TocTime, TicTime);
}

typedef vector<GeneticAlgorithmn> GAvector;

class Table {

public:

	Layer SingleSp;

};

typedef vector<Table> TableVector;

class CoEvolution: public NeuralNetwork,
		public virtual TrainingExamples,
		public GeneticAlgorithmn //,public virtual RandomNumber
{

public:

	int NoSpecies;
	GAvector Species;
	bool end;
	TableVector TableSp;

	vector<bool> NotConverged;
	Sizes SpeciesSize;
	Layer Individual;
	Data TempTable;
	int TotalEval;
	int TotalSize;
	int SingleGAsize;
	double Train;
	double Test;
	int kid;

	CoEvolution() {

	}

	void MainProcedure(bool bp, int RUN, ofstream &out1, ofstream &out2,
			ofstream &out3, double mutation, int depth);

	void InitializeSpecies();
	void EvaluateSpecies(NeuralNetwork network, TrainingExamples Sample);
	void GetBestTable(int sp);
	void PrintSpecies();

	void Join();
	double ObjectiveFunc(Layer x);
	void Print();

	void sort(int s);

	bool EndTraining() {
		return end;
	}
	;

	void find_parents(int s, NeuralNetwork network, TrainingExamples Sample);

	void EvalNewPop(int pass, int s, NeuralNetwork network,
			TrainingExamples Sample);

	void rep_parents(int s, NeuralNetwork network, TrainingExamples Sample);

	void EvolveSubPopulations(int repetitions, double h, NeuralNetwork network,
			TrainingExamples Sample, double mutation, int depth,
			ofstream &out1);

};

void CoEvolution::InitializeSpecies() {
	end = false;

	for (int Sp = 0; Sp < NoSpecies; Sp++) {
		Species.push_back(0);
	}

	for (int Sp = 0; Sp < NoSpecies; Sp++) {
		NotConverged.push_back(true);
	}

	TotalSize = 0;
	for (int row = 0; row < NoSpecies; row++)
		TotalSize += SpeciesSize[row];

	for (int row = 0; row < TotalSize; row++)
		Individual.push_back(0);

	for (int s = 0; s < NoSpecies; s++) {
		Species[s].NumVariable = SpeciesSize[s];
		Species[s].InitilisePopulation();
	}

	TableSp.resize(NoSpecies);

	for (int row = 0; row < NoSpecies; row++)
		for (int col = 0; col < SpeciesSize[row]; col++)
			TableSp[row].SingleSp.push_back(0);
}

void CoEvolution::PrintSpecies() {

	for (int s = 0; s < NoSpecies; s++) {
		Species[s].PrintPopulation();
		cout << s << endl;
	}
}

void CoEvolution::GetBestTable(int CurrentSp) {
	int Best;
	for (int sN = 0; sN < CurrentSp; sN++) {
		Best = Species[sN].BestIndex;

		// cout<<Best<<endl;
		for (int s = 0; s < SpeciesSize[sN]; s++)
			TableSp[sN].SingleSp[s] = Species[sN].Population[Best].Chrome[s];
	}

	for (int sN = CurrentSp; sN < NoSpecies; sN++) {
		// cout<<"g"<<endl;
		Best = Species[sN].BestIndex;
		//cout<<Best<<" ****"<<endl;
		for (int s = 0; s < SpeciesSize[sN]; s++)
			TableSp[sN].SingleSp[s] = Species[sN].Population[Best].Chrome[s];
	}
}

void CoEvolution::Join() {
	int index = 0;

	for (int row = 0; row < NoSpecies; row++) {
		for (int col = 0; col < SpeciesSize[row]; col++) {

			Individual[index] = TableSp[row].SingleSp[col];
			index++;
		}

	}
}

void CoEvolution::Print() {

	for (int row = 0; row < NoSpecies; row++) {
		for (int col = 0; col < SpeciesSize[row]; col++) {
			cout << TableSp[row].SingleSp[col] << " ";
		}
		cout << endl;
	}
	cout << endl;

	for (int row = 0; row < TotalSize; row++)
		cout << Individual[row] << " ";
	cout << endl << endl;
}

void CoEvolution::EvaluateSpecies(NeuralNetwork network,
		TrainingExamples Sample) {

	for (int SpNum = 0; SpNum < NoSpecies; SpNum++) {

		GetBestTable(SpNum);

//---------make the first individual in the population the best

		for (int i = 0; i < Species[SpNum].NumVariable; i++)
			TableSp[SpNum].SingleSp[i] = Species[SpNum].Population[0].Chrome[i];

		Join();
		Species[SpNum].Population[0].Fitness = network.ForwardFitnessPass(
				Individual, Sample); //ObjectiveFunc(Individual);
		TotalEval++;

		Species[SpNum].BestFit = Species[SpNum].Population[0].Fitness;
		Species[SpNum].BestIndex = 0;
		// cout<<"g"<<endl;
		//------------do for the rest

		for (int PIndex = 0; PIndex < PopSize; PIndex++) {

			for (int i = 0; i < Species[SpNum].NumVariable; i++)
				TableSp[SpNum].SingleSp[i] =
						Species[SpNum].Population[PIndex].Chrome[i];

			Join();

			Species[SpNum].Population[PIndex].Fitness =
					network.ForwardFitnessPass(Individual, Sample); //
			TotalEval++;

			if ((MINIMIZE * Species[SpNum].BestFit)
					> (MINIMIZE * Species[SpNum].Population[PIndex].Fitness)) {
				Species[SpNum].BestFit =
						Species[SpNum].Population[PIndex].Fitness;
				Species[SpNum].BestIndex = PIndex;
			}

		}
		cout << SpNum << " -- " << endl;
	}

}

double CoEvolution::ObjectiveFunc(Layer x) {
	int i, j, k;
	double fit, sumSCH;

	fit = 0.0;

#ifdef ellip
	// Ellipsoidal function
	for(j=0;j<TotalSize;j++)
	fit+=((j+1)*(x[j]*x[j]));
#endif

#ifdef schwefel
	// Schwefel's function
	for(j=0; j<TotalSize; j++)
	{
		for(i=0,sumSCH=0.0; i<j; i++)
		sumSCH += x[i];
		fit += sumSCH * sumSCH;
	}
#endif

#ifdef rosen
	//Rosenbrock's function
	for (j = 0; j < TotalSize - 1; j++)
		fit += 100.0 * (x[j] * x[j] - x[j + 1]) * (x[j] * x[j] - x[j + 1])
				+ (x[j] - 1.0) * (x[j] - 1.0);

	NumEval++;
#endif

	return (fit);
}

void CoEvolution::find_parents(int s, NeuralNetwork network,
		TrainingExamples Sample) //here the parents to be replaced are added to the temporary sub-population to assess their goodness against the new solutions formed which will be the basis of whether they should be kept or not
		{
	int i, j, k;
	double u, v;

	Species[s].my_family();

	for (j = 0; j < family; j++) {

		Species[s].NewPop[Species[s].kids + j].Chrome =
				Species[s].Population[Species[s].mom[j]].Chrome;

		GetBestTable(s);
		for (int i = 0; i < Species[s].NumVariable; i++)
			TableSp[s].SingleSp[i] =
					Species[s].NewPop[Species[s].kids + j].Chrome[i];
		Join();

		Species[s].NewPop[Species[s].kids + j].Fitness =
				network.ForwardFitnessPass(Individual, Sample); //ObjectiveFunc(Individual);
		TotalEval++;
	}
}

void CoEvolution::EvalNewPop(int pass, int s, NeuralNetwork network,
		TrainingExamples Sample) {

	GetBestTable(s);
	for (int i = 0; i < Species[s].NumVariable; i++)
		TableSp[s].SingleSp[i] = Species[s].NewPop[pass].Chrome[i];
	Join();

	Species[s].NewPop[pass].Fitness = network.ForwardFitnessPass(Individual,
			Sample); // ObjectiveFunc(Individual);
	TotalEval++;
}

void CoEvolution::sort(int s)

{
	int i, j, temp;
	double dbest;

	for (i = 0; i < (Species[s].kids + family); i++)
		Species[s].list[i] = i;

	if (MINIMIZE)
		for (i = 0; i < (Species[s].kids + family - 1); i++) {
			dbest = Species[s].NewPop[Species[s].list[i]].Fitness;
			for (j = i + 1; j < (Species[s].kids + family); j++) {
				if (Species[s].NewPop[Species[s].list[j]].Fitness < dbest) {
					dbest = Species[s].NewPop[Species[s].list[j]].Fitness;
					temp = Species[s].list[j];
					Species[s].list[j] = Species[s].list[i];
					Species[s].list[i] = temp;
				}
			}
		}
	else
		for (i = 0; i < (Species[s].kids + family - 1); i++) {
			dbest = Species[s].NewPop[Species[s].list[i]].Fitness;
			for (j = i + 1; j < (Species[s].kids + family); j++) {
				if (Species[s].NewPop[Species[s].list[j]].Fitness > dbest) {
					dbest = Species[s].NewPop[Species[s].list[j]].Fitness;
					temp = Species[s].list[j];
					Species[s].list[j] = Species[s].list[i];
					Species[s].list[i] = temp;
				}
			}
		}
}

void CoEvolution::rep_parents(int s, NeuralNetwork network,
		TrainingExamples Sample) //here the best (1 or 2) individuals replace the family of parents
		{
	int i, j;
	for (j = 0; j < family; j++) {

		Species[s].Population[Species[s].mom[j]].Chrome =
				Species[s].NewPop[Species[s].list[j]].Chrome;

		GetBestTable(s);

		for (int i = 0; i < Species[s].NumVariable; i++)
			TableSp[s].SingleSp[i] =
					Species[s].Population[Species[s].mom[j]].Chrome[i];
		Join();

		Species[s].Population[Species[s].mom[j]].Fitness =
				network.ForwardFitnessPass(Individual, Sample); //ObjectiveFunc(Individual);
		TotalEval++;
	}
}

void CoEvolution::EvolveSubPopulations(int repetitions, double h,
		NeuralNetwork network, TrainingExamples Samples, double mutation,
		int depth, ofstream &out1) {
	double tempfit;
	int count = 0;
	int tag;
	kid = KIDS;
	int numspecies = 0;

	for (int s = 0; s < NoSpecies; s++) {
		if (NotConverged[s] == true) {
			for (int r = 0; r < repetitions; r++) {

				tempfit = Species[s].Population[Species[s].BestIndex].Fitness;
				Species[s].kids = KIDS;

				Species[s].RandomParents();

				for (int i = 0; i < Species[s].kids; i++) {
					tag = Species[s].GenerateNewPCX(i, network, Samples,
							mutation, depth); //generate a child using PCX

					if (tag == 0) {
						NotConverged[s] = false;
						out1 << "tag1" << endl;
						break;
					}
				}
				if (tag == 0) {
					out1 << "tag2" << endl;
					NotConverged[s] = false;
				}

				for (int i = 0; i < Species[s].kids; i++)
					EvalNewPop(i, s, network, Samples);

				find_parents(s, network, Samples); // form a pool from which a solution is to be
				//   replaced by the created child

				Species[s].sort(); // sort the kids+parents by fitness
				//       sort(s);
				rep_parents(s, network, Samples); // a chosen parent is replaced by the child

				Species[s].BestIndex = 0;

				tempfit = Species[s].Population[0].Fitness;

				for (int i = 1; i < PopSize; i++)
					if ((MINIMIZE * Species[s].Population[i].Fitness)
							< (MINIMIZE * tempfit)) {
						tempfit = Species[s].Population[i].Fitness;
						Species[s].BestIndex = i;
					}

			}

		} //r

	} //species

}

class CombinedEvolution: public CoEvolution // ,      public virtual  NeuralNetwork, public virtual TrainingExamples, public virtual GeneticAlgorithmn
{

public:
	int TotalEval;
	int TotalSize;
	double Train;
	double Test;
	double TrainNMSE;
	double TestNMSE;
	double Error;
	CoEvolution NeuronLevel;
	CoEvolution WeightLevel;
	CoEvolution OneLevel;
	int Cycles;
	bool Sucess;

	CombinedEvolution() {

	}

	int GetEval() {
		return TotalEval;
	}
	double GetCycle() {
		return Train;
	}
	double GetError() {
		return Test;
	}

	double NMSETrain() {
		return TrainNMSE;
	}
	double NMSETest() {
		return TestNMSE;
	}

	bool GetSucess() {
		return Sucess;
	}

	void Procedure(bool bp, double h, ofstream &out1, ofstream &out2,
			ofstream &out3, double mutation, double depth);
};

void CombinedEvolution::Procedure(bool bp, double h, ofstream &out1,
		ofstream &out2, ofstream &out3, double mutation, double depth) {

	clock_t start = clock();

	int hidden = h;

	int output = 1;
	int input = 1;

	int weightsize1 = (input * hidden);

	int weightsize2 = (hidden * output);

	int contextsize = hidden * hidden;
	int biasize = hidden + output;

	const int outputsize = output;
	const int acousticVector = input;

	ofstream out;
	out.open("out.txt");
	int gene = 1;
	double trainpercent = 0;
	double testpercent = 0;
	int epoch;
	double testtree;

	char file[15] = "Learnt.txt";
	TotalEval = 0;
	double H = 0;

	TrainingExamples Samples(trainfile, trainsize, acousticVector, outputsize);
	//Samples.printData();

	double error;

	Sizes layersize;
	layersize.push_back(acousticVector);
	layersize.push_back(hidden);
	layersize.push_back(outputsize);

	NeuralNetwork network(layersize);
	network.CreateNetwork(layersize, trainsize);

	cout << " doing-----" << endl;

	if (bp) {
		epoch = network.BackPropogation(Samples, 0.2, layersize, file,
				trainfile, trainsize, acousticVector, outputsize); //  train the network

		cout << Train << "  " << Test << endl;
	}

	else {
		Sucess = false;

		Cycles = 0;

		for (int n = 0; n < hidden; n++)
			NeuronLevel.SpeciesSize.push_back(input + 1);

		for (int n = 0; n < hidden; n++)
			NeuronLevel.SpeciesSize.push_back(hidden);

		for (int n = 0; n < output; n++)
			NeuronLevel.SpeciesSize.push_back(hidden + 1);

		NeuronLevel.NoSpecies = hidden + hidden + output;
		NeuronLevel.InitializeSpecies();

#ifdef neuronlevel
		NeuronLevel.EvaluateSpecies(network, Samples);
#endif
		cout << h << ' ' << NeuronLevel.NoSpecies << endl;
		for (int s = 0; s < NeuronLevel.NoSpecies; s++)
			NeuronLevel.NotConverged[s] = true;

		cout << " Evaluated neuronLevel ----------->" << endl;
		//-----------------------------------------------

		TotalEval = 0;
		NeuronLevel.TotalEval = 0;

		int count = 0;

		//----------------------------------------------------

		bool end = false;

#ifdef neuronlevel
		while ((TotalEval <= maxgen) && (!end)) {
			NeuronLevel.EvolveSubPopulations(1, 1, network, Samples, mutation,
					0, out1);

			NeuronLevel.GetBestTable(NeuronLevel.NoSpecies - 1);
			NeuronLevel.Join();
			end = NeuronLevel.EndTraining();
			network.ChoromesToNeurons(NeuronLevel.Individual);
			network.SaveLearnedData(layersize, file);
			Error =
					NeuronLevel.Species[NeuronLevel.NoSpecies - 1].Population[NeuronLevel.Species[NeuronLevel.NoSpecies
							- 1].BestIndex].Fitness;
			//    if(count%30==0)
			out1 << hidden << " nl " << Train << "    " << Error << "    "
					<< NeuronLevel.TotalEval << "    " << count << endl;

			TotalEval = NeuronLevel.TotalEval;

			if (Error < MinimumError) {
				Sucess = true;
				break;
			}
			count++;

		}

#endif

		out2 << "Train" << endl;
		Train = network.TestTrainingData(layersize, file, trainfile, trainsize,
				acousticVector, outputsize, out2);
		TrainNMSE = network.NMSError();
		out2 << "Test" << endl;
		Test = network.TestTrainingData(layersize, file, testfile, testsize,
				acousticVector, outputsize, out2);
		TestNMSE = network.NMSError();
		out2 << endl;
		cout << Test << " was test RMSE " << endl;
		out1 << endl;
		out1 << " ------------------------------ " << h << "  " << TotalEval
				<< "  RMSE:  " << Train << "  " << Test << " NMSE:  "
				<< TrainNMSE << " " << TestNMSE << endl;

		out2 << " ------------------------------ " << h << "  " << TotalEval
				<< "  " << Train << "  " << Test << endl;
		out3 << "  " << h << "  " << TotalEval << "  RMSE:  " << Train << "  "
				<< Test << " NMSE:  " << TrainNMSE << " " << TestNMSE << endl;

	}

}

//---------------------------------------------------------------------------------------
int main(void) {
 
	ofstream out1;
	out1.open("Oneout1.txt");
	ofstream out2;
	out2.open("Oneout2.txt");
	ofstream out3;
	out3.open("Oneout3.txt");

	for (int hidden = 3; hidden <= 7; hidden += 2) {
		double onelevelstop = 0.05;
		double maxrun = 1; // choose number of experimental runs
		int success = 0;

		for (int run = 1; run <= maxrun; run++) {
			CombinedEvolution Combined;

			Combined.Procedure(false, hidden, out1, out2, out3, 0,
					onelevelstop);

			if (Combined.GetSucess()) {
				success++;
			}

		}

	} //hidden
	out1.close();
	out2.close();
	out3.close();

	return 0;

}
;
