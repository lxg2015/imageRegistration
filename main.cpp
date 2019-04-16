#include "opencv2/opencv.hpp"
#include "match.h"
#include <string>

using namespace cv;
using namespace std;

int main(int argc, char **argv){
    Match match;
    // match.load_tpl("../img/object0001.view02.png");
    // match.load_src("../img/object0001.view03.png");
    match.load_tpl("../img/02.jpg");
    match.load_src("../img/03.jpg");

    match.compute_simi();
    
    Mat img;
    match.transform_back(img);
    imwrite("../img/dst.png", img);
}