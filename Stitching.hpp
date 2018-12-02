//
//  Stitching.hpp
//  CV-5-again
//
//  Created by Kluni on 2018/11/28.
//  Copyright © 2018 Kluni. All rights reserved.
//

#ifndef Stitching_hpp
#define Stitching_hpp

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <set>
#include <map>
#include <queue>
#include <algorithm>
#include "CImg.h"
#include "Sift.h"

using namespace std;
using namespace cimg_library;

const float scale = 0.25;

class Stitching {
private:

    CImg<float> preprocessing(CImg<float> image);
    CImg<float> RGB2Gary(CImg<float> image);
public:
//    void test(string path, vector<string> names);
    void stitch(string DirPath);
protected:
};


#endif /* Stitching_hpp */
