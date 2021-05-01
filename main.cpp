#include "EnvSim.h"
#include "Eigen/Dense"
#include "fstream"
#include "Env.h"

int main()
{
	//return 0;
	srand(time(NULL));
	//..................

	//make LSTM instance
	LSTM* NeuralNet = new LSTM(40, 81, 30, 0.01, 10, 3, 1, true);
	//retrieve weights first
	NeuralNet->RetrieveAllWeights();
	//make Env instance
	Env* Environment = new Env();
	
	//loop EntireRun for how ever long and every certain amount turn on board view
	bool Show = false;
	for (int i = 0; i < 100; i++)
	{
		for (int i = 0; i < 50; i++)
		{
			if (i % 10 == 0)
			{
				Show = true;
				if (NeuralNet->Epsilon < 15)
				{
					NeuralNet->Epsilon = 15;
				}
			}
			std::cout << i << std::endl;
			Environment->EntireRun(30, 81, NeuralNet, Show);
			Show = false;
		}
		//Show = true;
		std::cout << "SAVED ALL WEIGHTS" << std::endl;
		NeuralNet->DeleteAllWeights(); //need to clear the file first
		NeuralNet->SaveAllWeights();
	}

	//after all that, save the weights
	NeuralNet->DeleteAllWeights(); //need to clear the file first
	NeuralNet->SaveAllWeights();
	
	return 0;
}
