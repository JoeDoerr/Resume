#include "LSTM.h"
#include "Eigen/Dense"
#include "iostream"
#include "math.h"
#include "stdlib.h"     /* srand, rand */
#include "time.h"       /* time */
#include "NN.h"
#include "array"

//Type constants
#define LEAKYRELUAF 0
#define TANHAF 1
#define SIGMOIDAF 2

#define NOACTIONCHOOSE -1
#define EPSILONSUBACTIONTYPE 0
#define EPSILONNORMACTIONTYPE 1

//Summary of entire LSTM:
//when epoch starts initialize memcell and rinput and set memoryiterator back to 0
//Have a memory size equal to epoch size
//Calculate gradients right after each run but don't put them in instantly (Use UpdateAndCleanNNs to apply them)
//Feed Forward remembers all the information for backprop at that timestep
//Backprop takes the error from a timestep and BPTTs it, can do errors from as many timesteps as want
//Has choose action functionality that has different modes of decision making. (Set a public variable to the action iterator we chose so some other functionality can use it)

//*Note: RInput must be the same size as ActionAmount

//...Setting matrices or arrays equal to each other means using: m2 = m1.cast<float>();
//Essentially if we using MatrixXd do not use anything with Xf because they cannot convert from float to double or the other way

LSTM::LSTM(int RInputSize, int InputSize, int MemSize, float LR, int AmountOfActions, int TruncateBP, int ChooseActionType, bool UsingCEC) :
	RInputSize(RInputSize), InputSize(InputSize), MemSize(MemSize), AmountOfActions(AmountOfActions), TruncateBP(TruncateBP), ChooseActionType(ChooseActionType), UsingCEC(UsingCEC)
{
	FinalInputSize = RInputSize + InputSize;
	//All NN's with 1 input, 1 hidden, 1 output, with input being rinput+input size, and output being rinput size, all same memory is necessary
	BlockInputNN = new NN(3, FinalInputSize, FinalInputSize, RInputSize, TANHAF, LEAKYRELUAF, LR, MemSize, 0, 0, 0, 0, 0, false, false, "BlockInput.txt");
	ForgetGateNN = new NN(3, FinalInputSize, FinalInputSize, RInputSize, SIGMOIDAF, LEAKYRELUAF, LR, MemSize, 0, 0, 0, 0, 0, false, false, "ForgetGate.txt");
	InputGateNN = new NN(3, FinalInputSize, FinalInputSize, RInputSize, SIGMOIDAF, LEAKYRELUAF, LR, MemSize, 0, 0, 0, 0, 0, false, false, "InputGate.txt");
	OutputGateNN = new NN(3, FinalInputSize, FinalInputSize, RInputSize, SIGMOIDAF, LEAKYRELUAF, LR, MemSize, 0, 0, 0, 0, 0, false, false, "OutputGate.txt");
	ActionChoosingNN = new NN(4, RInputSize, RInputSize, AmountOfActions, SIGMOIDAF, LEAKYRELUAF, 0.01, MemSize, 0, 0, RInputSize, 0, 0, false, true, "ActionChoosingNN.txt");

	//Scale initialized weights based on input amount
	BlockInputNN->InitializeWeightsScaledToInputAmount();
	ForgetGateNN->InitializeWeightsScaledToInputAmount();
	InputGateNN->InitializeWeightsScaledToInputAmount();
	OutputGateNN->InitializeWeightsScaledToInputAmount();

	MemoryIterator = 0;

	//Make info matrices
	for (int i = 0; i < MemSize; i++)
	{
		BlockOutMem[i] = Eigen::ArrayXXd(RInputSize, 1);
		InputOutMem[i] = Eigen::ArrayXXd(RInputSize, 1);
		ForgetOutMem[i] = Eigen::ArrayXXd(RInputSize, 1);
		OutputOutMem[i] = Eigen::ArrayXXd(RInputSize, 1);
		BlockOutMemPreAF[i] = Eigen::ArrayXXd(RInputSize, 1);
		InputOutMemPreAF[i] = Eigen::ArrayXXd(RInputSize, 1);
		ForgetOutMemPreAF[i] = Eigen::ArrayXXd(RInputSize, 1);
		OutputOutMemPreAF[i] = Eigen::ArrayXXd(RInputSize, 1);
		MemCellMem[i] = Eigen::ArrayXXd(RInputSize, 1);
	}
	
	InitializedMemCellMem = Eigen::ArrayXXd(RInputSize, 1);

	RInput = Eigen::MatrixXd(RInputSize, 1); //don't need to remember inputs at every step if we use the constant error carousel/also the NN's remember their states themselves
	MemCell = Eigen::ArrayXXd(RInputSize, 1); //Formatting like this means they can be used as inputs to NNs
}

//+ adds the matrices elementwise
//Arrays can be * if they have same size and do elementwise multiplication

void LSTM::InitializeBeforeStarting() //initialize rinput and memcell
{
	//Set memory iterator back to 0
	MemoryIterator = 0;

	//memcell randomized initialization
	MemCell = Eigen::ArrayXXd::Random(RInputSize, 1);
	InitializedMemCellMem = MemCell;

	//rinput all 0's initialization
	RInput = Eigen::MatrixXd::Zero(RInputSize, 1);
}

//Summary: Feeds into forget gate, updates memcell, feeds into blockinput and input gate, updates memcell, tanhmemcell, output gate, choose action, save RInput for next one, memit++
void LSTM::FeedForwardLSTM(Eigen::MatrixXd InputMatrix) //need to save the outputted values somewhere at each timestep
{
//recieve input and rinput and concatenate
	Eigen::MatrixXd RealInput(FinalInputSize, 1); //we want the input values to NN's to be vertical, so the columns stay constant and the up and down of rows changes
	for (int i = 0; i < InputSize; i++)
	{
		RealInput(i, 0) = InputMatrix(i, 0); //Put in the input values
	}
	for (int i = InputSize; i < FinalInputSize; i++)
	{
		int L = i - InputSize;
		RealInput(i, 0) = RInput(L, 0); //Put in the rinput values
	}
	
	//std::cout << "RealInput: " << RealInput << std::endl;
	
//update memcell with elementwise mutliplcation of forget gate's result
	ForgetGateNN->FeedForward(RealInput); //error here with the real input
	
	//std::cout << "ForgetGateNN Output: " << ForgetGateNN->ArrayOutN << std::endl;
	//Save forgetoutput in memory
	ForgetOutMemPreAF[MemoryIterator] = ForgetGateNN->OutN.array(); //OutN is the preAf version of the output
	ForgetOutMem[MemoryIterator] = ForgetGateNN->ArrayOutN; //Normal version
	
	MemCell *= ForgetGateNN->ArrayOutN; //two arrays elementwise multiplied
	
	//std::cout << "MemCell: " << MemCell << std::endl;

//make block input and elementwise multiply to input gate
	BlockInputNN->FeedForward(RealInput);
	InputGateNN->FeedForward(RealInput);
	//Save their outputs in memory
	
	//std::cout << "BlockInputNN Output: " << BlockInputNN->OutN << std::endl;
	//std::cout << "InputGateNN Output: " << InputGateNN->OutN << std::endl;

	BlockOutMemPreAF[MemoryIterator] = BlockInputNN->OutN.array(); //preaf
	BlockOutMem[MemoryIterator] = BlockInputNN->ArrayOutN; //normal

	InputOutMemPreAF[MemoryIterator] = InputGateNN->OutN.array();
	InputOutMem[MemoryIterator] = BlockInputNN->ArrayOutN;

//elementwise add scaled block input to scaled memcell
	MemCell += BlockInputNN->ArrayOutN * InputGateNN->ArrayOutN;
	//std::cout << "Block input elementwise multiplied to input gate: " << std::endl << BlockInputNN->ArrayOutN * InputGateNN->ArrayOutN;
	//std::cout << "MemCell: " << MemCell << std::endl;

//tanh this sum of block input and memcell AND //Save this MemCell in memory
	MemCellMem[MemoryIterator] = MemCell;
	for (int i = 0; i < RInputSize; i++)
	{
		MemCell(i, 0) = NN::Tanh(MemCell(i, 0));
	}

//Now factor in the Output gate
	//first run output gate
	OutputGateNN->FeedForward(RealInput);

	//Save memory
	OutputOutMemPreAF[MemoryIterator] = OutputGateNN->OutN.array(); //OutN is the preAf version of the output
	OutputOutMem[MemoryIterator] = OutputGateNN->ArrayOutN; //Normal version
	
	//elementwise multiply memcell by output gate
	MemCell *= OutputGateNN->ArrayOutN;

//Make RInput this output
	RInput = MemCell.matrix();

//output a decision (The mem cell is used as sort of the value being passed through)
	if (ChooseActionType != NOACTIONCHOOSE)
	{
		ChooseAction();
	}

//Save RInput as memory (the output of this timestep)
	BlockOutputMem[MemoryIterator] = MemCell;

//Update the timestep
	MemoryIterator++;
}

//The 3 steps to finding partial derivatives are:
		//freeze the world at the time
		//whatever you are changing, don't include their value (ex: want to change weights, look at the error neuron value and the neuron value)
		//include the derivative of the AF

//put in anything for memcell errors for when just beginning backprop
void LSTM::BackpropagateLSTM(Eigen::ArrayXXd Error, int CurrentTimestep, Eigen::ArrayXXd MemCellErrors, int CurrentRecursiveIteration, bool CEC) //unrolling just for one of the timestep's decisions at a time
{
	//std::cout << "Backprop at timestep: " << CurrentRecursiveIteration << std::endl;
	//Recieve the final value of success for action of this timestep as (changing, 1)
	//Run recursively all the way through unrolled LSTM, with only calculating the OutputGate errors once and just rerunning with the constant error carousel (CEC)

//Create all post output gate errors
	
	//The scaled error that is recycled and used in everything
	Eigen::ArrayXXd ScaledError(RInputSize, 1);

//For output gate gradients, the errors to give to the NN are found by:
	
	//freezing the world at that time expect for our output gate NN
	//So, elementwise multiply the output errors by the MemCell
	//Now that it knows how much of an effect there was on the errors, now multiply by dersigmoid(outputgatevalueattime)
	//Ex: If one of the errors wants it's value to go down, it will have some like -0.2 error. Mulitplied by 0.7 which is part of what got it there,
	//Ex cont: it scale that the output gate has a considerable amount of effect here and change that part of the output gate more than if the value was lower, something like 0.3
	if (CEC == false) //we only update the output gate if we are on the first run
	{
		//std::cout << "MemCellMemsize: " << MemCellMem[CurrentTimestep].size() << " Errorsize: " << Error.size() << " DerSigmoidGroupsize: " << NN::DerSigmoidGroup(OutputOutMemPreAF[CurrentTimestep]).size() << std::endl;
		ScaledError = Error * MemCellMem[CurrentTimestep] * NN::DerSigmoidGroup(OutputOutMemPreAF[CurrentTimestep]); //scale by memcell and dersigmoid of the output at the time
		
		OutputGateNN->LearnSingleRunThrough(ScaledError, CurrentTimestep); //now have the output gate update with these errors (error needs to be (number, 0)
		//also we truncate the error so the input errors are not used for anything rn at least
	}
	
//For memcell errors:

	//We freeze the world as it was, so elementwise multiply the output errors by the output gate outputs, then by the dertanh(memcellvalueattime)
	//What directions and magnitudes do we want the memcell to be to help the output be closer to what we want?
	if (CEC == false) //we only calculate the original MemCell error once, the rest of it is calculated from forget gate changes which happens when passing the MemCellErrors back
	{
		MemCellErrors = Eigen::ArrayXXd(RInputSize, 1);
		MemCellErrors = Error * OutputOutMem[CurrentTimestep] * NN::DerTanhGroup(MemCellMem[CurrentTimestep]); //set the memcellerrors this
	}
	
//For Input Gate errors:

	//Freeze the block input values, elementwise multiply the errors of the memcell by those to scale, then also do the dersigmoid for the values at the time for input gate, and those will be the output errors
	ScaledError = MemCellErrors * BlockOutMem[CurrentTimestep] * NN::DerSigmoidGroup(InputOutMemPreAF[CurrentTimestep]);
	InputGateNN->LearnSingleRunThrough(ScaledError, CurrentTimestep);

//For block input errors:
	//Freeze input gate values, elementwise multiply the errors of the memcell by those to scale, then also do the dertanh for the blockinput values at the time, then will be output errors
	ScaledError = MemCellErrors * InputOutMem[CurrentTimestep] * NN::DerTanhGroup(BlockOutMemPreAF[CurrentTimestep]);
	BlockInputNN->LearnSingleRunThrough(ScaledError, CurrentTimestep);

//For forget gate errors:

	//Freeze the elementwise multiplied previous timestep memcell to then elementwise multiply to scale the current memcell errors, then do dersigmoud for the forget gate values of the time, for output errors
	//Scaled with this timestep's errors, but past timestep's memcell scaled it's effect on now's timestep memcell, as well as its dersigmoid for itself at the time
	if (CurrentTimestep > 0) //when its 0 we can't do past timestep, we need the specific value of the initialized memcell
	{
		ScaledError = MemCellErrors * MemCellMem[CurrentTimestep - 1] * NN::DerSigmoidGroup(ForgetOutMemPreAF[CurrentTimestep]);
	}
	else
	{
		ScaledError = MemCellErrors * InitializedMemCellMem * NN::DerSigmoidGroup(ForgetOutMemPreAF[CurrentTimestep]);
	}
	ForgetGateNN->LearnSingleRunThrough(ScaledError, CurrentTimestep);

//Recursively run and finish on TruncateBp so wherever we want to be stopping

	//std::cout << "Current Timestep: " << CurrentTimestep << std::endl;
	CurrentRecursiveIteration++;
	if (CurrentRecursiveIteration < TruncateBP && CurrentTimestep - 1 >= 0)
	{
		//First update memcellerrors:
		
		MemCellErrors *= ForgetOutMem[CurrentTimestep]; //scale by forgetout for the next backprop because the MemCell is heavily influenced by previous memcells in feedforward,
		//so in backprop the memcells ahead of them want to edit past memcells as part of their error. But to do that, they are scaled by their forget gate at the time, so that needs to be calced
		
		BackpropagateLSTM(Error, CurrentTimestep - 1, MemCellErrors, CurrentRecursiveIteration, UsingCEC); //if we have usingcec as true, then we will be running CEC, if not, then it runs without CEC
	}
}

void LSTM::RunBackprop(float Rewards[], int EpochLength)
{
	//develop the error at each iterator then backprop it
	for (int i = 0; i < EpochLength; i++)
	{
		//float ActionError = Rewards[i] - OutputOutMem[i](ActionMemory[i], 0);
		float ActionError = Rewards[i] - ActionChoosingNN->OutputMemory(i, ActionMemory[i]);
		
		Eigen::MatrixXd Errors = Eigen::MatrixXd::Zero(AmountOfActions, 1);
		
		//if they have a negative reward, optimize all other actions only up to 1/5 of what outputoutmem was
		if (ActionError < 0)
		{
			//float OptimalValue = OutputOutMem[i](ActionMemory[i], 0) / 5; //1/5 the value that the failed action had
			float OptimalValue = ActionChoosingNN->OutputMemory(i, ActionMemory[i]) / 5;
			for (int b = 0; b < AmountOfActions; b++)
			{
				if (ActionChoosingNN->OutputMemory(i, b) < OptimalValue) //only do if it is to increase it
				{
					Errors(b, 0) = OptimalValue - ActionChoosingNN->OutputMemory(i, b);
				}
			}
		}
		else if(ActionError > 0) //if just a positive change, optimize others down to 0, and if it is exactly the same (=), then errors are all 0!
		{
			for (int b = 0; b < AmountOfActions; b++)
			{
				//Errors(b, 0) = 0 - OutputOutMem[i](b, 0);
				Errors(b, 0) = 0 - ActionChoosingNN->OutputMemory(i, b);
			}
		}
		
		//Then put in the action error into the errors
		Errors(ActionMemory[i], 0) = ActionError;
		
		//now use these errors to backproagate ActionChoosingNN, then the input errors of ActionChoosingNN will be used on LSTMBackprop
		ActionChoosingNN->BackPropagate(i, Errors);
		
		//now finally we can do 
		BackpropagateLSTM(ActionChoosingNN->InputError, i, Errors); //remember you can put anything for memcell when backprop cuz it will be made themselves
		
	}
	UpdateAndCleanNNs();
}

void LSTM::UpdateAndCleanNNs()
{
	//update the NN's then set the gradients back to 0
	BlockInputNN->ApplyGradients();
	BlockInputNN->CleanGradients();

	ForgetGateNN->ApplyGradients();
	ForgetGateNN->CleanGradients();

	InputGateNN->ApplyGradients();
	InputGateNN->CleanGradients();

	OutputGateNN->ApplyGradients();
	OutputGateNN->CleanGradients();

	ActionChoosingNN->ApplyGradients();
	ActionChoosingNN->CleanGradients();
}

void LSTM::ChooseAction()
{
	ActionChoosingNN->FeedForward(MemCell, false);

	//Finding the highest value output
	float Highest = -10000;
	int It = -1;
	//Highest Value
	for (int i = 0; i < AmountOfActions; i++)
	{
		if (ActionChoosingNN->ArrayOutN(i, 0) > Highest)
		{
			Highest = ActionChoosingNN->ArrayOutN(i, 0);
			It = i;
		}
	}
	//if we have an error where none of the outputs satisfy this
	if (It == -1)
	{
		std::cout << "...Choose action iterator = -1! " << MemCell << std::endl;
	}
	
	
	//Here we are doing different methods of action choosing based on our chooseactiontype
	//We return the iterator of the action we chose after putting it in the mem, we output it so we can do things based on the action we just chose
	if (ChooseActionType == EPSILONSUBACTIONTYPE)
	{
		ActionChosenIterator = It;
	}
	else if (ChooseActionType == EPSILONNORMACTIONTYPE)
	{
		if (0 == rand() % Epsilon)
		{
			It = rand() % AmountOfActions;
		}
		ActionChosenIterator = It;
	}

	ActionMemory[MemoryIterator] = ActionChosenIterator;

	//std::cout << "At this memoryit: " << MemoryIterator << " Chose this action: " << ActionChosenIterator << std::endl;
}

void LSTM::RetrieveAllWeights()
{
	BlockInputNN->RetrieveWeights();
	InputGateNN->RetrieveWeights();
	ForgetGateNN->RetrieveWeights();
	OutputGateNN->RetrieveWeights();
	ActionChoosingNN->RetrieveWeights();
}

void LSTM::SaveAllWeights()
{
	BlockInputNN->SaveWeights();
	InputGateNN->SaveWeights();
	ForgetGateNN->SaveWeights();
	OutputGateNN->SaveWeights();
	ActionChoosingNN->SaveWeights();
}

void LSTM::DeleteAllWeights()
{
	BlockInputNN->DeleteWeights();
	InputGateNN->DeleteWeights();
	ForgetGateNN->DeleteWeights();
	OutputGateNN->DeleteWeights();
	ActionChoosingNN->DeleteWeights();
}
