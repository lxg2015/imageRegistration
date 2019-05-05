#include "tool.h"
#include <fstream>
#include <algorithm>
#include <iostream>
#include <string>

using namespace std;

void load_img_list(string name, vector<string> &img_list){
    ifstream in(name.c_str());
    string line;
    if(in.is_open()){
        while(getline(in, line)){
            img_list.push_back(line);
        }
    }
}

void load_config(string name, map<string, float> &mv){
    ifstream in(name.c_str());
    string line;
    if(in.is_open()){
        while(getline(in, line)){
            line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
            if(line[0] == '#' || line.empty()){
                continue;
            }
            size_t pos = line.find("=");
            string name = line.substr(0, pos);
            string value = line.substr(pos + 1);
            mv[name] = std::stof(value);
        }
    }
}