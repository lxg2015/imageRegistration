#include "tool.h"
#include <fstream>

using namespace std;

void load_img_list(string name, vector<string> &img_list){
    ifstream in(name.c_str());
    string line;

    while(getline(in, line)){
        img_list.push_back(line);
    }
}