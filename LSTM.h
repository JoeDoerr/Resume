#pragma once
#include "Eigen/Dense"
#include "NN.h"
#include "array"

#define MAXMEM 200

class LSTM
{
public:
	LSTM(int RInputSize, int InputSize, int MemSize, float LR, int AmountOfActions, int TruncateBP, int ChooseActionType, bool UsingCEC);

	//Consists of block input, input gate, forget gate, output gate
	//I will have the constructor indicate what gates we want, the size of the input and the rinput

	void InitializeBeforeStarting(); //Initialize the RInput and MemCell

	void FeedForwardLSTM(Eigen::MatrixXd InputMatrix); //Run through once, needing to call function again to go to the next timestep

	void BackpropagateLSTM(Eigen::ArrayXXd Error, int CurrentTimestep, Eigen::ArrayXXd MemCellErrors, int CurrentRecursiveIteration = 0, bool CEC = false); //Backpropagate unrolled LSTM from the current timestep's errors
	//call again for different timestep errors to be used

	//make an int array for each reward in the epoch then pass in here to do automatic backprop for the epoch
	void RunBackprop(float Rewards[], int EpochLength);

	void UpdateAndCleanNNs();

	int ActionChosenIterator = 0;

	std::array<Eigen::ArrayXXd, MAXMEM> BlockOutputMem;
	std::array<Eigen::ArrayXXf, 200> BlockOutputMe;

	int Epsilon = 5; //want epsilon to be externally changed

	//retrieve the whole LSTM of neural nets
	void RetrieveAllWeights();
	void SaveAllWeights();
	void DeleteAllWeights();

	//to use the LSTM, know the epoch length you want and instantiate it with that as memory, can just put in inputs each time into FeedForward then take the actionchoseniterator out and use that
	//for backprop make the "Error" yourself with what 

private:

	//Functions//
	void ChooseAction();
	NN* ActionChoosingNN;

	//Size Values:
	const int RInputSize = 0;
	const int InputSize = 0;
	int FinalInputSize = 0;
	const int MemSize = 0;
	const int AmountOfActions = 0;

	//Other Variables
	const int ChooseActionType = 0;

	const int TruncateBP = 0;

	const bool UsingCEC = false;

	int MemoryIterator = 0;

	std::array<int, MAXMEM> ActionMemory; //makes backprop with RL easier, //records action iterators

	Eigen::MatrixXd RInput;
	Eigen::ArrayXXd MemCell;

	//Neural Networks
	NN* BlockInputNN;
	NN* ForgetGateNN;
	NN* InputGateNN;
	NN* OutputGateNN;


	//Memory Array Matrices:

	//All will be matrices of timestep and size, all being the rinput size
	//Pieces of the LSTM needing to be saved:
	 //PreAF block output to calculate Input Gate derivative (PreAF so we know the deraf and have something to plug into it), Also the actual output
		//Eigen::MatrixXd BlockOutMemPreAF;
		//Eigen::MatrixXd BlockOutMem;
	std::array<Eigen::ArrayXXd, MAXMEM> BlockOutMemPreAF; //keep the amount of arrays larger than we will ever have memory and we won't be iterating through it all
	std::array<Eigen::ArrayXXd, MAXMEM> BlockOutMem;
	 //PreAf input gate to calculate Block Input derivative
		//Eigen::MatrixXd InputOutMemPreAF;
		//Eigen::MatrixXd InputOutMem;
	std::array<Eigen::ArrayXXd, MAXMEM> InputOutMemPreAF;
	std::array<Eigen::ArrayXXd, MAXMEM> InputOutMem;
	 //PreAf forget gate to calculate itself with the deraf
		//Eigen::MatrixXd ForgetOutMemPreAF;
		//Eigen::MatrixXd ForgetOutMem;
	std::array<Eigen::ArrayXXd, MAXMEM> ForgetOutMemPreAF;
	std::array<Eigen::ArrayXXd, MAXMEM> ForgetOutMem;
	 //PreAf output gate to calculate the memcell rate of change and itself with deraf
		//Eigen::MatrixXd OutputOutMemPreAF;
		//Eigen::MatrixXd OutputOutMem;
	std::array<Eigen::ArrayXXd, MAXMEM> OutputOutMemPreAF;
	std::array<Eigen::ArrayXXd, MAXMEM> OutputOutMem;
	 //MemCell state at the time
		//Eigen::MatrixXd MemCellMem;  //PreTanh Elementwise added memcell after forget gate with the scaled blockinput * input gate value to find the dertanh of the time (This is the memcell)
	std::array<Eigen::ArrayXXd, MAXMEM> MemCellMem;
	Eigen::ArrayXXd InitializedMemCellMem;
};

//..Some logic..\\
//MemCell state at the time
	  //something adding to the memcell does not change how the memcell affects the error, it simply contributes to what the memcell's value is
	  //The value is added then the memcell does it's effects, so this is a prior change. To then see how the additive values could contribute to changing the memcell's values,
	  //we want to show how much their change will affect the memcell. That means that even if a million things were added alongside, that won't change how their individual change will affect the memcell
	 
