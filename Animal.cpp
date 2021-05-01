#include "Animal.h"
#include "Env.h"
#include "Eigen/Dense"
#include "iostream"
#include "math.h"
#include "stdlib.h"     /* srand, rand */
#include "time.h"       /* time */

Animal::Animal(int XX, int YY)
{
	X = XX;
	Y = YY;
}

void Animal::Move(Env* Prey)
{
	if (ExhaustionLevel >= 6 && Haste > 0)
	{
		ExhaustionLevel = 0;
		Haste--;
		return;
	}
	ExhaustionLevel++;
	//whichever value the abs difference of x's and y's is the direction being moved, then if neg left, pos right, diagonals too!
	
	int UpDown = Prey->Y - Y; //positive means Y++, neg means Y--;
	int RightLeft = Prey->X - X; //same but for X, if prey is at x=4 and im at x=1, then 4-1 is positive 3, x++ will help me get there
	int MoveY = 1;
	int MoveX = 1;
	if (UpDown < 0)
	{
		MoveY = -1;
	}
	if (RightLeft < 0)
	{
		MoveX = -1;
	}

	if (abs(UpDown) > abs(RightLeft))
	{
		if (Prey->CheckWallCollision(X, Y + MoveY) == true)
		{
			Y += MoveY;
		}
	}
	else if (abs(UpDown) < abs(RightLeft))
	{
		if (Prey->CheckWallCollision(X + MoveX, Y) == true)
		{
			X += MoveX;
		}
	}
	else //equal
	{
		if (Prey->CheckWallCollision(X + MoveX, Y + MoveY) == true)
		{
			Y += MoveY;
			X += MoveX;
		}
	}

	if (X == Prey->BombX && Y == Prey->BombY)
	{
		Haste = 30;
		Prey->BombX = -1;
		Prey->BombY = -1;
	}
}

