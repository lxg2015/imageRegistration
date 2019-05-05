#ifndef _TOOL_H_
#define _TOOL_H_

#include <string>
#include <vector>
#include <map>

void load_img_list(std::string name, std::vector<std::string> &img_list);
void load_config(std::string name, std::map<std::string, float> &mv);

#endif