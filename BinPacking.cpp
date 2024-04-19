// BinPacking.cpp : définit le point d'entrée de l'application.
//

#include "BinPacking.h"
#include <fstream>

using namespace std;

const string DATA_PATH = "./data/";

int main()
{
	fstream file;
	file.open("test.txt", ios::in);

	cout << "Hello CMake." << endl;
	return 0;
}
