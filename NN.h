#pragma once
#include "Eigen/Dense"
#include "string"
//#include "cstdint"

//using namespace Eigen;
//Always for matrices: (Prev Hidden layer neurons, Next Hidden layer neurons)
//(Rows, Columns)
//-> Rows     \/ columns
//So in one PrevHiddenLayerNeuron row, there is 1 of each NextHiddenLayerNeuron through columns
//This NN set up can do 6 layers

//Layer includes input and output
class NN
{
public:
	NN(int LayerAmount, int NI, int NH1, int NOut, int ActFuncOut, int ActFuncHidden, double LR = 0.01, int MemorySize = 100, 
		int GetRewardType = 0, int ChooseActionType = 0, int NH2 = 0, int NH3 = 0, int NH4 = 0, bool ActionChooserNN = false, bool SaveOutputs = false, std::string FileName = "weforgottoputfilename");
	
	//Memory storage://

	//(MemoryIt, Iterator for values in that memory)
	Eigen::MatrixXd StateMemory;

	//(Memory It, Iterator for values in the output)
	Eigen::MatrixXd OutputMemory;

	//What ended up being the value of the output actions currently
	Eigen::MatrixXd TempOutputMemory;

	//What action did we actually do
	Eigen::MatrixXd OutputChosenMemory;

	//(MemoryIt, 0)
	Eigen::MatrixXd RewardMemory;


	int MemoryIt = 0;

	//End Memory Storage//


	//true if we don't recalculate for the outputs in backprop, false would be for when we do backprop out of order
	bool SaveOutputs = false;

	//GetRewardType == 0
	int SuccessfulAction = 0;

	//for if we want to track rewards each step real time
	float SuccessAmount = 0;
	int RunTimes = 1;

	//Initialize weights correctly so that the NN doesn't start off with exploding values
	void InitializeWeightsScaledToInputAmount();

private:

	//activation function trackers
	int ActFunctionOut = 0; //.. 0 is LeakyReLU, 1 is Tanh, 2 is sigmoid
	int ActFunctionHidden = 0; //.. 0 is LeakyReLU, 1 is Tanh

	//constants
	const int Layers = 1;
	const int Inputs = 0;
	const int MemorySize = 100;
	const int GetRewardType = 0;
	const int ChooseActionType = 0; //0 is epsilon sub, 1 is normal epsilon
	const bool ActionChooserNN = false; //Is this a neural network that chooses actions itself
	const std::string FileName; //save the weights after we finish something

	//misc variables
	int Epsilon = 5;
	double LR = 0.01;

	//to more easily figure the last hidden layer to have go into the output weight matrix
	int LastHiddenSize = 0;
	Eigen::MatrixXd LastHiddenL;
	void GetLastHiddenL();

	//NN architecture
	const int NI;
	const int NH1;
	const int NH2;
	const int NH3;
	const int NH4;
	const int NOut;

	//Weights (Malleable pointers set in CreateWeights) (The First is input to h1, the the next ones can be h1 to h2 stuff or hsomething to output)
	//Eigen::MatrixXd FirstW;
	Eigen::MatrixXd SecondW;
	Eigen::MatrixXd ThirdW;
	Eigen::MatrixXd FourthW;
	Eigen::MatrixXd OutW;

	//Backprop variables (D for delta, change in) (These are to save the changes we want to make while waiting for a batch to be done)
	Eigen::MatrixXd FirstW_D;
	Eigen::MatrixXd SecondW_D;
	Eigen::MatrixXd ThirdW_D;
	Eigen::MatrixXd FourthW_D;
	Eigen::MatrixXd OutW_D;

	//Neuron values
	//Eigen::MatrixXd InputN;
	Eigen::MatrixXd FirstN;
	Eigen::MatrixXd SecondN;
	Eigen::MatrixXd ThirdN;
	Eigen::MatrixXd FourthN;
	//OutN is public

	//Biases
	Eigen::MatrixXd FirstB;
	Eigen::MatrixXd SecondB;
	Eigen::MatrixXd ThirdB;
	Eigen::MatrixXd FourthB;
	Eigen::MatrixXd OutB;

	//Backprop variables (D for delta, change in) (These are to save the changes we want to make while waiting for a batch to be done)
	Eigen::MatrixXd FirstB_D;
	Eigen::MatrixXd SecondB_D;
	Eigen::MatrixXd ThirdB_D;
	Eigen::MatrixXd FourthB_D;
	Eigen::MatrixXd OutB_D;

	//Functions//
	void SoftmaxOutputs();

public:
	Eigen::MatrixXd FirstW;

	//Neuron Value output is public so it can be used
	Eigen::ArrayXXd ArrayOutN; //post activation functioned output
	Eigen::MatrixXd OutN; //pre activation functioned output

	//InputError
	Eigen::MatrixXd InputError;

	///static functions:///
	static Eigen::MatrixXd Concatenate(Eigen::MatrixXd FirstMatrix, Eigen::MatrixXd SecondMatrix, int FirstSize, int SecondSize);

	//activation functions
	static double LeakyReLU(double In);
	static double Tanh(double In);
	static double Sigmoid(double In);

	static double DerLeakyReLU(double In);
	static double DerTanh(double In);
	static Eigen::ArrayXXd DerTanhGroup(Eigen::ArrayXXd In);
	static Eigen::ArrayXXd DerSigmoidGroup(Eigen::ArrayXXd In);
	static double DerSigmoid(double In);

	//Explosive Gradient Stopper
	void ScaleGradients();
	const double DesiredMean = 0.02;

	double ActivationFunction(int IsHiddenLayer, double In, bool UsingDerivative = false);

	//Functions that need to be accessable by main function
	void FeedForward(Eigen::MatrixXd Input, bool ForBackprop = false);
	//Use the Neuron Value Matrices in backprop
	void BackPropagate(int MemIt, Eigen::MatrixXd Error);
	//Choose action
	int ChooseAction();
	//Get Rewards
	void GetRewardsEachStep(int ActionChosen);

	//For LSTMs
	 //After the run, calculate and save the gradients then later put them in
	 //The value at each timestep has to be saved in memory
	 //Error values are propagated back to each of these functions from the other, so this takes in a matrix
	void LearnSingleRunThrough(Eigen::MatrixXd Errors, int MemIt);
	void CleanGradients();
	void ApplyGradients();

	void RetrieveWeights();
	void SaveWeights();
	void DeleteWeights();
};
