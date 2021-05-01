#pragma once

class Env;

class Animal
{
public:
	Animal(int XX, int YY);

	int X = 1;
	int Y = 1;

	void Move(Env* Prey);

	int ExhaustionLevel = 0;
	int Haste = 3;
	

private:

};
