#include "NN.h"
#include "Eigen/Dense"
#include "iostream"
#include "math.h"
#include "stdlib.h"     /* srand, rand */
#include "time.h"       /* time */

using namespace Eigen;

#define EPSILONSUBACTIONTYPE 0
#define EPSILONNORMACTIONTYPE 1

#define EACHSTEPREWARDTYPE 0

#define LEAKYRELUAF 0
#define TANHAF 1
#define SIGMOIDAF 2
#define SOFTMAXAF 3

//2 layer means just the input and output, 3 layers means output and 1 hidden, and so forth
NN::NN(int LayerAmount, int NI, int NH1, int NOut, int ActFuncOut, int ActFuncHidden, int BatchSize, bool UsingRandomizedLearning, double LR, int MemorySize, int GetRewardType, int ChooseActionType, int NH2, int NH3, int NH4, bool ActionChooserNN)
	: Inputs(NI), Layers(LayerAmount), NI(NI), NH1(NH1), NOut(NOut), NH2(NH2), NH3(NH3), NH4(NH4), LR(LR), MemorySize(MemorySize), UsingRandomizedLearning(UsingRandomizedLearning), BatchSize(BatchSize),
	GetRewardType(GetRewardType), ChooseActionType(ChooseActionType), ActionChooserNN(ActionChooserNN)
{
	StateMemory = MatrixXd(MemorySize, NI);
	TempOutputMemory = MatrixXd(NOut, 1);
	OutputChosenMemory = MatrixXd(MemorySize, 1);
	RewardMemory = MatrixXd(MemorySize, 1);
	
	ActFunctionOut = ActFuncOut;
	ActFunctionHidden = ActFuncHidden;

	//Get first hidden size to help smooth out backprop
	if (NH4 != 0)
	{
		LastHiddenSize = NH4;
	}
	else if (NH3 != 0)
	{
		LastHiddenSize = NH3;
	}
	else if (NH2 != 0)
	{
		LastHiddenSize = NH2;
	}
	else
	{
		LastHiddenSize = NH1;
	}

	int RecentLayer = 0;
	for (int i = 0; i < Layers; i++)
	{
		//While we aren't on the last one, we keep setting more hidden layers
		if (i != Layers - 1)
		{
			//[NextLayer, PastLayer]
			switch (i)
			{
			//Input Layer, no biases
			case 0:
				//InputN = MatrixXd(NI, 1);
				break;
			//H1 Layer, biases, weight from past to this
			case 1:
				FirstN = MatrixXd(NH1, 1);
				FirstB = MatrixXd::Random(NH1, 1);
				FirstW = MatrixXd::Random(NH1, NI);
				
				//Backprop updates:
				FirstB_D = MatrixXd::Zero(NH1, 1);
				FirstW_D = MatrixXd::Zero(NH1, NI);

				RecentLayer = NH1;
				break;
			//H2 Layer, biases, weight from past to this
			case 2:
				SecondN = MatrixXd(NH2, 1);
				SecondB = MatrixXd::Random(NH2, 1);
				SecondW = MatrixXd::Random(NH2, NH1);

				//Backprop updates:
				SecondB_D = MatrixXd::Zero(NH2, 1);
				SecondW_D = MatrixXd::Zero(NH2, NH1);

				RecentLayer = NH2;
				break;
			//H3 Layer, biases, weight from past to this
			case 3:
				ThirdN = MatrixXd(NH3, 1);
				ThirdB = MatrixXd::Random(NH3, 1);
				ThirdW = MatrixXd::Random(NH3, NH2);

				//Backprop updates:
				ThirdB_D = MatrixXd::Zero(NH3, 1);
				ThirdW_D = MatrixXd::Zero(NH3, NH2);

				RecentLayer = NH3;
				break;
			//H4 Layer, biases, weight from past to this
			case 4:
				FourthN = MatrixXd(NH4, 1);
				FourthB = MatrixXd::Random(NH4, 1);
				FourthW = MatrixXd::Random(NH4, NH3);

				//Backprop updates:
				FourthB_D = MatrixXd::Zero(NH4, 1);
				FourthW_D = MatrixXd::Zero(NH4, NH3);

				RecentLayer = NH4;
				break;
			default:
				break;
			}
		}
		else if (i == Layers - 1)
		{
			OutN = MatrixXd(NOut, 1);
			ArrayOutN = ArrayXXd(NOut, 1);
			OutB = MatrixXd::Random(NOut, 1);
			OutW = MatrixXd::Random(NOut, RecentLayer);

			//Backprop updates:
			OutB_D = MatrixXd::Zero(NOut, 1);
			OutW_D = MatrixXd::Zero(NOut, LastHiddenSize);
		}
	}
}

//Initialize weights correctly so that the NN doesn't start off with exploding values
void NN::InitializeWeightsScaledToInputAmount()
{
	//Amount 10 inputs is generally standard good for this so all the weights can be divided by: (input num / 10)
	float ScalingFactor = (float)NI / 20.0;

	int RecentLayer = 0;
	for (int i = 0; i < Layers; i++)
	{
		if (i != Layers - 1)
		{
			//[NextLayer, PastLayer]
			switch (i)
			{
				//Input Layer, no biases
			case 0:
				break;
			case 1:
				FirstB /= ScalingFactor;
				FirstW /= ScalingFactor;
				RecentLayer = NH1;
				break;
				//H2 Layer, biases, weight from past to this
			case 2:
				SecondB /= ScalingFactor;
				SecondW /= ScalingFactor;
				RecentLayer = NH2;
				break;
				//H3 Layer, biases, weight from past to this
			case 3:
				ThirdB /= ScalingFactor;
				ThirdW /= ScalingFactor;
				RecentLayer = NH3;
				break;
				//H4 Layer, biases, weight from past to this
			case 4:
				FourthB /= ScalingFactor;
				FourthW /= ScalingFactor;
				break;
			default:
				break;
			}
		}
		else if (i == Layers - 1)
		{
			OutB /= ScalingFactor;
			OutW /= ScalingFactor;
		}
	}
}

//(row, column)
//(NextLayer, PastLayer)
//For the hidden layers, we have rows filled up
//The weights look like [row-> column\/]
///////////////////////////////////////
///////////////////////////////////////
//[ROW FOR OUTPUTS, COLUMN FOR INPUTS]
///////////////////////////////////////
///////////////////////////////////////

//Access specific elements of matrices use:
//MatrixName(0, 1);

//When input is matrix multiplied to layer weights, each iteration value of the input is (if the input is one column) will be multiplied to each row value in the weight's column of that iteration
//That is how each input value applies itself to each output value. Each row takes each column in its row and sums it to create an output vector.
//[a1, a2] [I1] = [(I1*a1) + (I2*a2)]
//[b1, b2] [I2]   [(I1*b1) + (I2*b2)]
//[c1, c2]        [(I1*c1) + (I2*c2)]

//For matrices, it is each column is given to the corresponding iterator of the inputs going in
//So for the neuron0, their weights to their respective next layer hidden neuron is in column0


//Summary of FeedForward:
//First fill in the state memory if we aren't feeding forward for backprop
//Then for each layer that we have:
  //Matrix multiply layer weights with past layer outputs or the initial inputs
  //Add biases to resulting vector
  //apply activation function
  //Save this layer as the most recently done layer
//Calculate the output layer at the end using most recently done layer's output values as the input
//If this is for backpropagation, we record the output values temporarily
//If this is for decision making, we choose an action and can do other various functionalities
//After this, if this is not for backpropataion increment the memory iterator

void NN::FeedForward(Eigen::MatrixXd Input, bool ForBackprop)
{
	//std::cout << "FeedForward: " << std::endl;
	//First put the inputs into the input slot

	//When just data gathering is true, we are just grabbing the output to then backprop with, so we are filling up a temp output array to learn with only, don't need state mem
	if (ForBackprop == false)
	{
		for (int i = 0; i < Inputs; i++)
		{
			//then save the inputs in memory
			//(everything that is like a neuron layer or input layer or output layer needs to be this vertical thing of 1 column, with values going down the row)
			StateMemory(MemoryIt, i) = Input(i, 0); //since we are iterating through it doesn't matter if we are holding data vertically or horizontally, statemem can deposit it vertically if it needs
		}
	}

	//have a way to remember the hidden layer outputs temporarily
	//Answer: just use the hidden layer neuron values

	int RecentMatrix = 0;

	//Now matrix multiply down the line:
	for (int i = 0; i < Layers; i++)
	{
		if (i != Layers - 1)
		{
			//So for how many layers we have, we continue until we reach the end then do the output
			switch (i)
			{
			case 0:
				//Do nothing here, its just acknowledging that the input layer exists
				break;
			case 1:
				//On the 1st hidden layer, matrix multiply the inputs with their weights to the H1 Layer
				FirstN = FirstW * Input;
				//Add the biases
				FirstN = FirstN + FirstB;
				//std::cout << "FirstN: " << FirstN << std::endl;

				//Activation function (NextLayer, PastLayer)
				for (int j = 0; j < NH1; j++)
				{
					FirstN(j, 0) = NN::ActivationFunction(0, FirstN(j, 0));
				}
				//std::cout << "First hidden layer: " << std::endl << FirstN << std::endl;
				RecentMatrix = 1;
				break;
			case 2:
				//First hidden layer to the second hidden layer
				SecondN = SecondW * FirstN;
				//Add the biases
				SecondN = SecondN + SecondB;
				//Activation function (NextLayer, PastLayer)
				for (int j = 0; j < NH2; j++)
				{
					SecondN(j, 0) = NN::ActivationFunction(0, SecondN(j, 0));
				}
				RecentMatrix = 2;
				break;
			case 3:
				//Second hidden layer to third hidden layer
				ThirdN = ThirdW * SecondN;
				//Add the biases
				ThirdN = ThirdN + ThirdB;
				//Activation function (NextLayer, PastLayer)
				for (int j = 0; j < NH3; j++)
				{
					ThirdN(j, 0) = NN::ActivationFunction(0, ThirdN(j, 0));
				}
				RecentMatrix = 3;
				break;
			case 4:
				//Second hidden layer to third hidden layer
				FourthN = FourthW * ThirdN;
				//Add the biases
				FourthN = FourthN + FourthB;
				//Activation function (NextLayer, PastLayer)
				for (int j = 0; j < NH4; j++)
				{
					FourthN(j, 0) = NN::ActivationFunction(0, FourthN(j, 0));
				}
				RecentMatrix = 4;
				break;
			default:
				break;
			}
		}
		else if (i == Layers - 1)
		{
			//Recent Hidden layer to output layer
			switch (RecentMatrix)
			{
			case 0:
				OutN = OutW * Input;
			case 1:
				OutN = OutW * FirstN;
				break;
			case 2:
				OutN = OutW * SecondN;
				break;
			case 3:
				OutN = OutW * ThirdN;
				break;
			case 4:
				OutN = OutW * FourthN;
				break;
			default:
				break;
			}

			//std::cout << "OutN: " << OutN << std::endl;

			//Add the biases
			OutN = OutN + OutB;
			//Activation function (NextLayer, PastLayer)
			//If not softmax
			if (ActFunctionOut != SOFTMAXAF)
			{
				for (int j = 0; j < NOut; j++)
				{
					ArrayOutN(j, 0) = NN::ActivationFunction(1, OutN(j, 0));
				}
			}
			else
			{
				//softmax
				SoftmaxOutputs();
			}
			//std::cout << "OutN: " << OutN << std::endl;
			//std::cout << "Post AF OutN with this activation function: activation function: " << ActFunctionOut << " Here is Post AF OutN: " << ArrayOutN << std::endl;
		}
	}

	//if we are doing this for backprop, then we are saving a temp output memory
	if (ForBackprop == true)
	{
		for (int j = 0; j < NOut; j++)
		{
			TempOutputMemory = ArrayOutN;
			//TempOutputMemory(j, 0) = OutN(j, 0);
		}
	}
	else if (ActionChooserNN == true)
	{
		//Do the action based on what the output was and then also increment the memory after things are saved
		int ChosenAction = ChooseAction();
		//for rewards at each step, it is getrewardtype == 0
		if (GetRewardType == EACHSTEPREWARDTYPE)
		{
			GetRewardsEachStep(ChosenAction);
		}
		//if we don't do eachsteprewardtype, then we just don't get rewards here at all, it will come later
	}

	if (ForBackprop == false)
	{
		MemoryIt++;
		if (MemoryIt >= MemorySize)
		{
			MemoryIt = 0;
		}
	}
}

int NN::ChooseAction()
{
	std::cout << ArrayOutN << std::endl;

	//Finding the highest value output
	float h = -10000;
	int It = -1;
	//Highest Value
	for (int i = 0; i < NOut; i++)
	{
		if (ArrayOutN(i, 0) > h)
		{
			h = ArrayOutN(i, 0);
			It = i;
		}
	}
	//if we have an error where none of the outputs satisfy this
	if (It == -1)
	{
		std::cout << ".......................Choose action iterator = -1! " << ArrayOutN << std::endl;
	}
	OutputChosenMemory(MemoryIt, 0) = It;
	return It;


	//Here we are doing different methods of action choosing based on our chooseactiontype
	//We return the iterator of the action we chose after putting it in the mem, we output it so we can do things based on the action we just chose
	if (ChooseActionType == EPSILONSUBACTIONTYPE) //PNN won't need epsilon, should be going by generational learned NN to pick things
	{
		OutputChosenMemory(MemoryIt, 0) = It;
		return It;
	}
	else if (ChooseActionType == EPSILONNORMACTIONTYPE)
	{
		if (0 == rand() % Epsilon)
		{
			It = rand() % NOut;
		}
		OutputChosenMemory(MemoryIt, 0) = It;
		return It;
	}
}

//The successful action is a public variable that can be changed externally to be set to the correct action
void NN::GetRewardsEachStep(int ActionChosen)
{
	//std::cout << "MemoryIt " << MemoryIt << std::endl;
	RunTimes++;
	if (RunTimes >= 50)
	{
		RunTimes = 1;
		SuccessAmount = 0;
	}

	std::cout << SuccessfulAction << " <-SucAc, ChoAct-> " << ActionChosen << std::endl;

	//SuccessfulAction is setup in the Main.cpp
	if (SuccessfulAction == ActionChosen)
	{
		//Get reward memory here
		RewardMemory(MemoryIt, 0) = 1;
		SuccessAmount++;
		std::cout << RunTimes << " Success Rate: " << SuccessAmount / RunTimes << std::endl;
		return;
	}

	//If no, then we failed
	RewardMemory(MemoryIt, 0) = 0;
	std::cout << RunTimes << " Success Rate: " << SuccessAmount / RunTimes << std::endl;
}

void NN::GetLastHiddenL()
{
	if (NH4 != 0)
	{
		LastHiddenL = FourthN;
	}
	else if (NH3 != 0)
	{
		LastHiddenL = ThirdN;
	}
	else if (NH2 != 0)
	{
		LastHiddenL = SecondN;
	}
	else
	{
		LastHiddenL = FirstN;
	}
}

void NN::BackPropagate(int MemIt, Eigen::MatrixXd Error)
{
	//std::cout << "Backprop" << MemIt << std::endl;

	//Get the output:
	//First, make the input
	MatrixXd TempM(NI, 1);
	for (int i = 0; i < NI; i++)
	{
		TempM(i, 0) = StateMemory(MemIt, i);
	}
	//Temp output memory and all the hidden layer values are now saved
	//Now run a feedforward to have the hidden neuron values ready
	FeedForward(TempM, true);

	//Errors
	MatrixXd HLError = MatrixXd::Zero(LastHiddenSize, 1);
	//Use in each of the cases
	MatrixXd NextError;

	//Update the weights of hidden layer to outputs (Make sure this is hidden layer before the output)
	GetLastHiddenL();

	MatrixXd Temp = LastHiddenL; //...Pointer to Last Hidden Layer neuron matrix ex: FourthN
	for (int Out = 0; Out < NOut; Out++)
	{
		for (int LastH = 0; LastH < LastHiddenSize; LastH++)
		{
			////////////////              error of this layer * Previous Neuron Output * LR * derivative of activation function with respect to this current hidden layer's output (Out)
			OutW_D(Out, LastH) += Error(Out, 0) * Temp(LastH, 0) * LR * ActivationFunction(1, TempOutputMemory(Out, 0), true);

			//update the bias gradient for outputs
			OutB_D(Out, 0) += Error(Out, 0) * LR;

			//Now make the error for LastHidden
			////////////////     Error from Out * weight from LastHidden to Out
			HLError(LastH, 0) += Error(Out, 0) * OutW(Out, LastH);
		}
	}

	for (int i = Layers - 2; i > 0; i--) //starts at layers-2, ends at 1
	{
		switch (i)
		{
		case 4:
			//Weight update is the error of this layer * Previous Neuron Output * LR * derivative of activation function
			//Update H3 to H4 weights
			//[ROW FOR OUTPUTS, COLUMN FOR INPUTS]
			//The columns corresponds to which H3 Neuron we are looking at
			//The rows correspond to which H4 Neuron we are looking at
			//The error is a 1 dimensional matrix with just rows corresponding to neuron errors

			NextError = MatrixXd::Zero(NH3, 1);

			for (int H3 = 0; H3 < NH3; H3++)
			{
				for (int H4 = 0; H4 < NH4; H4++)
				{
					//Weight Gradients
					////////////////  error of this layer * Previous Neuron Output * LR * derivative of activation function with respect to this current hidden layer's output (H4)
					FourthW_D(H4, H3) += HLError(H4, 0) * ThirdN(H3, 0) * LR * ActivationFunction(0, FourthN(H4, 0), true);
					//outputs, inputs
					//H4, H3

					//Now make the error for H3
					////////////////    Error from H4 * weight from H3 to H4
					NextError(H3, 0) += HLError(H4, 0) * FourthW(H4, H3);
				}
			}
			
			for (int H4 = 0; H4 < NH4; H4++)
			{
				FourthB_D(H4, 0) += HLError(H4, 0) * LR;
			}

			HLError = NextError;

			break;
		case 3:
			//Update H2 to H3 weights
			NextError = MatrixXd::Zero(NH2, 1);

			for (int H2 = 0; H2 < NH2; H2++)
			{
				for (int H3 = 0; H3 < NH3; H3++)
				{
					//Weight Gradients
					////////////////  error of this layer * Previous Neuron Output * LR * derivative of activation function with respect to this current hidden layer's output (H3)
					ThirdW_D(H3, H2) += HLError(H3, 0) * SecondN(H2, 0) * LR * ActivationFunction(0, ThirdN(H3, 0), true);
					//outputs, inputs
					//H3, H2


					//Now make the error for H2
					////////////////    Error from H3 * weight from H2 to H3
					NextError(H2, 0) += HLError(H3, 0) * ThirdW(H3, H2);
				}
			}

			for (int H3 = 0; H3 < NH4; H3++)
			{
				ThirdB_D(H3, 0) += HLError(H3, 0) * LR;
			}

			HLError = NextError;

			//The weight changes are to make the more important neurons have more an effect on the correct neurons
			//The next error is to make the neuron understand how much of an effect is has had, and change it accordingly (magnitude and direction)

			break;
		case 2:
			//Update H1 to H2 weights
			NextError = MatrixXd::Zero(NH1, 1);
			
			for (int H1 = 0; H1 < NH1; H1++)
			{
				for (int H2 = 0; H2 < NH2; H2++)
				{
					//For checking:
					//std::cout << HLError(H2, 0) << " HLError(H2, 0)" << std::endl;
					//std::cout << FirstN(H1, 0) << " FirstN(H1, 0)" << std::endl;
					//std::cout << ActivationFunction(0, SecondN(H2, 0), true) << " ActivationFunction" << std::endl;

					//Weight Gradients
					////////////////  error of this layer * Previous Neuron Output * LR * derivative of activation function with respect to this current hidden layer's output (H2)
					SecondW_D(H2, H1) += HLError(H2, 0) * FirstN(H1, 0) * LR * ActivationFunction(0, SecondN(H2, 0), true);
					//outputs, inputs
					//H2, H1

					
					//Now make the error for H1
					////////////////    Error from H2 * weight from H1 to H2
					NextError(H1, 0) += HLError(H2, 0) * SecondW(H2, H1);
				}
			}

			for (int H2 = 0; H2 < NH2; H2++)
			{
				SecondB_D(H2, 0) += HLError(H2, 0) * LR;
			}

			HLError = NextError;

			break;
		case 1:
			//Update Input to H1 weights
			InputError = MatrixXd::Zero(NI, 1);

			for (int NIN = 0; NIN < NI; NIN++)
			{
				for (int H1 = 0; H1 < NH1; H1++)
				{
					//Weight Gradients
					////////////////  error of this layer * Previous Neuron Output * LR * derivative of activation function with respect to this current hidden layer's output (NI)
					FirstW_D(H1, NIN) += HLError(H1, 0) * FirstN(NIN, 0) * LR * ActivationFunction(0, FirstN(H1, 0), true);
					//outputs, inputs
					//H1, NI

					//Now make the error for IN
					////////////////    Error from H1 * weight from IN to H1
					InputError(NIN, 0) += HLError(H1, 0) * FirstW(H1, NIN);
				}
			}

			for (int H1 = 0; H1 < NH1; H1++)
			{
				FirstB_D(H1, 0) += HLError(H1, 0) * LR;
			}

			break;
		default:
			break;
		}
	}
}

void NN::CleanGradients()
{
	//Clean up the gradients first
	FirstB_D = MatrixXd::Zero(NH1, 1);
	FirstW_D = MatrixXd::Zero(NH1, NI);

	if (NH2 != 0)
	{
		SecondB_D = MatrixXd::Zero(NH2, 1);
		SecondW_D = MatrixXd::Zero(NH2, NH1);
	}
	if (NH3 != 0)
	{
		ThirdB_D = MatrixXd::Zero(NH3, 1);
		ThirdW_D = MatrixXd::Zero(NH3, NH2);
	}
	if (NH4 != 0)
	{
		FourthB_D = MatrixXd::Zero(NH4, 1);
		FourthW_D = MatrixXd::Zero(NH4, NH3);
	}

	OutB_D = MatrixXd::Zero(NOut, 1);
	OutW_D = MatrixXd::Zero(NOut, LastHiddenSize);
}

void NN::ApplyGradients()
{
	ScaleGradients();

	//std::cout << "Updating NN, the hidden to out weight gradients are: " << std::endl << OutW_D << std::endl;

	//After gradients are filled up, learn by adding all the biases and weights to their respective biases and weights
	if (NH1 != 0)
	{
		FirstB += FirstB_D;
		FirstW += FirstW_D;

		//std::cout << "Weight gradient input to first hidden layer: " << std::endl << FirstW_D << std::endl;
	}

	if (NH2 != 0)
	{
		SecondB += SecondB_D;
		SecondW += SecondW_D;
	}
	if (NH3 != 0)
	{
		ThirdB = ThirdB_D;
		ThirdW = ThirdW_D;
	}
	if (NH4 != 0)
	{
		FourthB = FourthB_D;
		FourthW = FourthW_D;
	}

	OutB += OutB_D;
	OutW += OutW_D;
}


//For LSTM--- takes the errors propagated to it then pits them against their output values to make the errors that will run through backprop
void NN::LearnSingleRunThrough(Eigen::MatrixXd Errors, int MemIt)
{
	//For all the NN's, they hit their inputs and calculate the error for our outputs because they saved our outputs in their StateMemory
	//So the error coming to this, is already our output error, we don't need to do anything else with our output.
	BackPropagate(MemIt, Errors);
}


void NN::ScaleGradients() //this turns out to already exist and is called gradient clipping
{
	double M = 0;
	int TimesAdded = 4;

	M += abs(FirstB_D.mean());
	M += abs(FirstW_D.mean());
	
	if (NH2 != 0)
	{
		M += abs(SecondB_D.mean());
		M += abs(SecondW_D.mean());
		TimesAdded += 2;
	}
	if (NH3 != 0)
	{
		M = abs(ThirdB_D.mean());
		M = abs(ThirdW_D.mean());
		TimesAdded += 2;
	}
	if (NH4 != 0)
	{
		M = abs(FourthB_D.mean());
		M = abs(FourthW_D.mean());
		TimesAdded += 2;
	}

	M += abs(OutB_D.mean());
	M += abs(OutW_D.mean());

	std::cout << "Mean: " << M << std::endl;
	if (M > DesiredMean)
	{
		//Value to multiply all values by:
		M = DesiredMean / M; //Mean * x = DesiredMean

		std::cout << "Multiplier: " << M << std::endl;

		//Multiply them all by the "X" (M)
		FirstB_D *= M;
		FirstW_D *= M;

		if (NH2 != 0)
		{
			SecondB_D *= M;
			SecondW_D *= M;
		}
		if (NH3 != 0)
		{
			ThirdB_D *= M;
			ThirdW_D *= M;
		}
		if (NH4 != 0)
		{
			FourthB_D *= M;
			FourthW_D *= M;
		}

		OutB_D *= M;
		OutW_D *= M;
	}
}

//Hidden layer = 0, Output = 1
//0 is LeakyReLU, 1 is Tanh
//0 is hidden layer, 1 is output
double NN::ActivationFunction(int IsHiddenLayer, double In, bool UsingDerivative)
{
	//std::cout << "Activation function: " << ActFunctionOut << std::endl;
	if (UsingDerivative == false)
	{
		//hidden layer
		if (IsHiddenLayer == 0)
		{
			if (ActFunctionHidden == LEAKYRELUAF)
			{
				return NN::LeakyReLU(In);
			}
			else if (ActFunctionHidden == TANHAF)
			{
				return NN::Tanh(In);
			}
		}
		//is output
		else
		{
			if (ActFunctionOut == LEAKYRELUAF)
			{
				return NN::LeakyReLU(In);
			}
			else if (ActFunctionOut == TANHAF)
			{
				return NN::Tanh(In);
			}
			else if (ActFunctionOut == SIGMOIDAF)
			{
				//std::cout << "Sigmoid af" << std::endl;
				return NN::Sigmoid(In);
			}
		}
	}
	else //using derivative of the activation function so return the derivative of the activation function
	{
		//hidden layer
		if (IsHiddenLayer == 0)
		{
			if (ActFunctionHidden == LEAKYRELUAF)
			{
				return NN::DerLeakyReLU(In);
			}
			else if (ActFunctionHidden == TANHAF)
			{
				return NN::DerTanh(In);
			}
		}
		//is output
		else
		{
			if (ActFunctionOut == LEAKYRELUAF)
			{
				return NN::DerLeakyReLU(In);
			}
			else if (ActFunctionOut == TANHAF)
			{
				return NN::DerTanh(In);
			}
			else if (ActFunctionOut == SIGMOIDAF)
			{
				return NN::DerSigmoid(In);
			}
		}
	}
	return 1.0;
}

double NN::LeakyReLU(double In)
{
	if (In > 0)
	{
		return In;
	}
	else
	{
		return In * 0.001;
	}
}

double NN::DerLeakyReLU(double In)
{
	if (In > 0)
	{
		return 1;
	}
	else
	{
		return 0.001;
	}
}

double NN::Tanh(double In)
{
	//std::cout << "Tanh output: " << tanh(In) << " Input into Tanh: " << In << std::endl;
	if (In > 10)
	{
		std::cout << "Tanh Exploding, Tanh output: " << tanh(In) << " Input into Tanh: " << In << std::endl;
	}
	return tanh(In);
}

double NN::DerTanh(double In)
{
	double sh = 1.0 / std::cosh(In);   // sech(x) == 1/cosh(x)
	return sh * sh;                     // sech^2(x)
}

Eigen::ArrayXXd NN::DerTanhGroup(Eigen::ArrayXXd In)
{
	for (int i = 0; i < In.size(); i++)
	{
		In(i, 0) = 1.0 / std::cosh(In(i, 0));
	}
	return In;
}

Eigen::ArrayXXd NN::DerSigmoidGroup(Eigen::ArrayXXd In)
{
	for (int i = 0; i < In.size(); i++)
	{
		In(i, 0) = (exp(-In(i, 0))) / pow((1 + exp(-In(i, 0))), 2);
	}
	return In;
}

double NN::Sigmoid(double In)
{
	return 1 / (1 + exp(-In));
}

double NN::DerSigmoid(double In)
{
	//(e^-x)/(1+e^-x)^2
	return (exp(-In)) / pow((1 + exp(-In)), 2);
}

void NN::SoftmaxOutputs() 
{
	//Put all the values as-> e^value
	MatrixXd Temp(NOut, 1);
	double Sum = 0;
	for (int i = 0; i < NOut; i++)
	{
		OutN(i, 0) = exp(OutN(i, 0));
		Sum += OutN(i, 0);
	}
	//Now we have the values to the e^value and we have the sum, now divide each for their final value
	for (int i = 0; i < NOut; i++)
	{
		OutN(i, 0) = OutN(i, 0) / Sum;
	}
}
