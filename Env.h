#pragma once
#include "Eigen/Dense"
#include "array"
#include "LSTM.h"

class Animal;

class Env
{
public:
	Env();

	//summary:
	//2d environment 16 x 16
	//moves up, down, left, right, and diagonals, and can just stay put so 9 options each step
	//chased by predators that move towards the agent then when in range lock on and move two steps at a time in that original direction it decided when it locked on

	//board things//
	const int BoardSize = 16;

	//character has one coordinate
	int X = BoardSize / 2;
	int Y = BoardSize / 2;

	//Running things//
	int MemoryIterator = 0;

	//does an entire game
	void EntireRun(int EpochLength, int InputSize, LSTM* Policy, bool Show = false);

	//method to move, automatically waste the agent's turn if walking into a wall or greater than 16 or less than 1, 1-16 values
	void AgentUpdateLocation(int ActionChosen);

	bool CheckWallCollision(int XCoord, int YCoord);

	//method to check everything's location, see if any xy values match up
	bool CheckAnimalCollisions();

	//make rewards
	void MakeRewards(LSTM* Policy);

	
	//player
	int BombX = -1;
	int BombY = -1;

	//cosmetic:
	void ShowBoard();

	const int AmountOfAnimals = 3;
	//std::array<Animal*, 3> AnimalList;
	Animal* AnimalList[3];
	//Animal* Animal1;
	//Animal* Animal2;
	//Animal* Animal3;

private:
	//blocks placed in the same formation
	const int AmountOfBlocks = 20;
	std::array<int, 20> BlocksX;
	std::array<int, 20> BlocksY;

	int WallXValue1;
	int WallXValue2;
};


