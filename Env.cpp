#include "Env.h"
#include "Eigen/Dense"
#include "iostream"
#include "math.h"
#include "LSTM.h"
#include "Animal.h"

//complete game summary each epoch:
//loop game for maximum timestep length
//state is entire board
//place walls and animals
//give the agent 3 timesteps to move before animals move
//give lstm current state then output an action
//use epsilon greedy to decide action
//move agent
//move animals then check for collisions
//if collision, game ends, if timesteps end game ends
//give reward based on turns survived
//backprop LSTM
//can show board when wanting to view game

Env::Env()
{
	///walls///
	WallXValue1 = BoardSize / 4;
	WallXValue2 = (BoardSize / 4) * 3;
	for (int i = 0; i < 10; i++)
	{
		BlocksX[i] = WallXValue1;
	}

	for (int i = 10; i < 20; i++)
	{
		BlocksX[i] = WallXValue2;
	}

	//straight line on x = 4
	BlocksY[0] = 11;
	BlocksY[1] = 10;
	BlocksY[2] = 9;
	BlocksY[3] = 8;
	BlocksY[4] = 7;
	BlocksY[5] = 6;
	BlocksY[6] = 5;
	BlocksY[7] = 4;
	BlocksY[8] = 3;
	BlocksY[9] = 2;

	//straight line on x = 12
	BlocksY[10] = 11;
	BlocksY[11] = 10;
	BlocksY[12] = 9;
	BlocksY[13] = 8;
	BlocksY[14] = 7;
	BlocksY[15] = 6;
	BlocksY[16] = 5;
	BlocksY[17] = 4;
	BlocksY[18] = 3;
	BlocksY[19] = 2;


	///animals simply spawning///
	for (int a = 0; a < AmountOfAnimals; a++)
	{
		AnimalList[a] = new Animal(a + 1, a + 1);
		//AnimalList[a]->Prey = this;
	}

	//Animal1 = new Animal(1, 1);
	//Animal1->Prey = this;
	//Animal2 = new Animal(5, 5);
	//Animal2->Prey = this;
	//Animal2 = new Animal(11, 11);
	//Animal2->Prey = this;
}

void Env::EntireRun(int EpochLength, int InputSize, LSTM* Policy, bool Show)
{
	//initializers
	MemoryIterator = 0;
	Policy->InitializeBeforeStarting();

	//reset the animals and agent positions
	X = BoardSize / 2;
	Y = BoardSize / 2;
	for (int a = 0; a < AmountOfAnimals; a++)
	{
		AnimalList[a]->Haste = 3;
		AnimalList[a]->X = (rand() % (BoardSize - 1)) + 1; //1 to 16
		AnimalList[a]->Y = (rand() % (BoardSize - 1)) + 1; //1 to 16
		while (AnimalList[a]->X == X && AnimalList[a]->Y == Y)
		{
			AnimalList[a]->X = (rand() % (BoardSize - 1)) + 1; //1 to 16
			AnimalList[a]->Y = (rand() % (BoardSize - 1)) + 1; //1 to 16
		}
	}

	

	for (int i = 0; i < EpochLength; i++)
	{
		//when we want to show the board
		if (Show == true)
		{
			ShowBoard();
		}
		
		//make the input
		Eigen::MatrixXd Input = Eigen::MatrixXd::Zero(InputSize, 1);
		
		//the x value of the coordinate is x * BoardSize and the y value is just y
		//(x * BoardSize) + y
		//then our own position is in another entire set of coordinates for the input
		//Input((X * 16) + Y + (BoardSize * BoardSize), 0) = 1;
		/*
		Input((X * 16) + Y, 0) = 1;
		
		for (int a = 0; a < AmountOfAnimals; a++) //animals
		{
			Input((AnimalList[a]->X * BoardSize) + AnimalList[a]->Y, 0) = 1;
		}

		for (int b = 0; b < AmountOfBlocks; b++) //blocks on the same coordinate info as animals
		{
			Input((BlocksX[b] * BoardSize) + BlocksY[b], 0) = 1;
		}
		*/
		

		//Input size is 81 blocks and is just the area surrounding agent
		//we simply see the bottom left from ourselves in the box as (0,0) so all coordinates will be animalx-XBottomLeft and animaly-YBottomLeft then put in as usual (Y*9)+X
		int XBottomLeft = X - 4;
		int YBottomLeft = Y - 4;
		//+4 all around, we are 9x9
		//if the item is > than x+4 or x-4 then not included
		for (int a = 0; a < AmountOfAnimals; a++) //animals
		{
			if (AnimalList[a]->X <= X + 4 && AnimalList[a]->X >= X - 4 && AnimalList[a]->Y <= Y + 4 && AnimalList[a]->Y >= Y - 4) //if we are in range
			{
				//we simply see ourselves as (0,0) so all coordinates will be animalx-X and animaly-Y then put in as usual (Y*9)+X
				Input(((AnimalList[a]->Y - YBottomLeft) * 9) + (AnimalList[a]->X - XBottomLeft), 0) = 1;
			}
		}
		for (int b = 0; b < AmountOfBlocks; b++) //blocks on the same coordinate info as animals
		{
			if (BlocksX[b] <= X + 4 && BlocksX[b] >= X - 4 && BlocksY[b] <= Y + 4 && BlocksY[b] >= Y - 4) //if we are in range
			{
				//Input(((BlocksY[b]- YBottomLeft) * 9) + (BlocksX[b] - XBottomLeft), 0) = 1;
			}
		}
		

		//choose action
		Policy->FeedForwardLSTM(Input);
		
		AgentUpdateLocation(Policy->ActionChosenIterator);
		//std::cout << "here" << std::endl;
		
		bool End = CheckAnimalCollisions();

		MemoryIterator++;

		if (End == true)
		{
			if (Show == true)
			{
				ShowBoard();
			}
			break;
		}
	}

	//when we succeed there is a glitch

	MakeRewards(Policy); //and backprops the LSTM, uses memory iterator for how long the epoch was
	if(MemoryIterator == 30)
	{
		std::cout << "SUCCESSSSS_________________________________________SUCCESS________________________________! * * * !";
	}
	std::cout << "Ended at Iterator: " << MemoryIterator << std::endl;
}

void Env::AgentUpdateLocation(int ActionChosen)
{
	switch (ActionChosen)
	{
	case 0: 
		//right
		if (CheckWallCollision(X + 1, Y) == true)
		{
			X++;
		}
		break;
	case 1:
		//left
		if (CheckWallCollision(X - 1, Y) == true)
		{
			X--;
		}
		break;
	case 2:
		//up
		if (CheckWallCollision(X, Y + 1) == true)
		{
			Y++;
		}
		break;
	case 3:
		//down
		if (CheckWallCollision(X, Y - 1) == true)
		{
			Y--;
		}
		break;
	case 4:
		//up right
		if (CheckWallCollision(X + 1, Y + 1) == true)
		{
			X++;
			Y++;
		}
		break;
	case 5:
		//up left
		if (CheckWallCollision(X - 1, Y + 1) == true)
		{
			X--;
			Y++;
		}
		break;
	case 6:
		//down right
		if (CheckWallCollision(X + 1, Y - 1) == true)
		{
			X++;
			Y--;
		}
		break;
	case 7:
		//down left
		if (CheckWallCollision(X - 1, Y - 1) == true)
		{
			X--;
			Y--;
		}
		break;
	case 8:
		//place bomb on self
		BombX = X;
		BombY = Y;
	//case 9 is when you do nothing
	default:
		break;
	}
}

bool Env::CheckWallCollision(int XCoord, int YCoord)
{
	if (XCoord > BoardSize || XCoord < 1 || YCoord > BoardSize || YCoord < 1)
	{
		return false;
	}
	else if (XCoord != WallXValue1 && XCoord != WallXValue2)
	{
		return true;
	}
	else
	{
		for (int i = 0; i < AmountOfBlocks; i++)
		{
			if (BlocksY[i] == YCoord && BlocksX[i] == XCoord)
			{
				return false;
			}
		}
	}

	return true;
}

bool Env::CheckAnimalCollisions()
{
	//let animals move
	for (int i = 0; i < AmountOfAnimals; i++)
	{
		AnimalList[i]->Move(this);

		//now check if the game is over
		if (AnimalList[i]->X == X && AnimalList[i]->Y == Y)
		{
			return true;
		}
	}
	return false;
}

void Env::MakeRewards(LSTM* Policy)
{
	//reward value based on epoch length so memory iterator value
	//later rewards, e^0.039x / 5 with full epoch length of 30 jumping to a 1

	float Arr[30]; //array can be bigger than needed, the values past when we stopped will never be used anyway
	if (MemoryIterator == 30)
	{
		for (int i = 0; i < 30; i++)
		{
			Arr[i] = 1;
		}
	}
	else if (MemoryIterator >= 10)
	{
		float Value = exp((float)MemoryIterator * 0.059) / 8;
		for (int i = 0; i < 30; i++)
		{
			Arr[i] = Value;
		}
	}
	else
	{
		//float Value = exp((float)MemoryIterator * 0.059) / 15;
		float Value = (float)MemoryIterator / 1900.0;
		for (int i = 0; i < 30; i++)
		{
			Arr[i] = Value;
		}
	}

	//Policy->Epsilon = 25;
	//epsilon updating, max is 25, lowest is 5, 15 steps is very bad, 30 is very good
	if (MemoryIterator > 20 && Policy->Epsilon < 25)
	{
		Policy->Epsilon++;
	}
	else if (MemoryIterator > 15 && Policy->Epsilon < 10)
	{
		Policy->Epsilon++;
	}
	else if(MemoryIterator < 8 && Policy->Epsilon > 6)
	{
		Policy->Epsilon--;
	}

	std::cout << "Make rewards running backprop" << std::endl;

	Policy->RunBackprop(Arr, MemoryIterator); //error here, I ended up doing 31 backprop at 
}

void Env::ShowBoard()
{
	int wdadwdada = 11;
	for (int i = 0; i < 100000000; i++) //exists so the board can type out slower and so we can actually see what happens in the game
	{
		wdadwdada = 11 + i / 2;
	}

	std::cout << std::endl << std::endl;
	for (int y = 0; y < BoardSize; y++)
	{
		std::cout << std::endl; //each y value we draw the whole line of x then when we go to the next one we endl
		for (int x = 0; x < BoardSize; x++)
		{
			std::cout << " ";
			char WhatSymbol = '0';
			for (int a = 0; a < AmountOfAnimals; a++)
			{
				if (AnimalList[a]->X == x && AnimalList[a]->Y == y)
				{
					WhatSymbol = '/';
					break;
				}
			}

			for (int b = 0; b < AmountOfBlocks; b++)
			{
				if (BlocksX[b] == x && BlocksY[b] == y)
				{
					WhatSymbol = '#';
				}
			}

			if (BombX == x && BombY == y)
			{
				WhatSymbol = 'B';
			}

			if (X == x && Y == y)
			{
				if (WhatSymbol == '/')
				{
					WhatSymbol = '*';
				}
				else
				{
					WhatSymbol = '^';
				}
			}
			std::cout << WhatSymbol;
		}
	}
}
