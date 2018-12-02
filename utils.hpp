//
//  utils.hpp
//  CV-5-again
//
//  Created by Kluni on 2018/12/1.
//  Copyright © 2018 Kluni. All rights reserved.
//

#ifndef utils_hpp
#define utils_hpp

#include <stdio.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <dirent.h>
#include "CImg.h"
#include "Sift.h"
#include "kdtree.h"
#include <set>
#include <time.h>
#include <stdlib.h>

using namespace std;
using namespace cimg_library;

#define NUM_OF_PAIR 4
#define CONFIDENCE 0.99
#define INLINER_RATIO 0.5
#define RANSAC_THRESHOLD 4.0

// 读取文件夹中全部的文件名
vector<string> ReadFilesFromDir(string DirPath);


// 进行球面投影
const float sphericalAngle = 22.0;
CImg<float> sphericalProjection(const CImg<float> &src);
float bilinear_interpolation(const CImg<float>& image, float x, float y, int channel);


// 关键点匹配

namespace Match{
    const float ratioThresh = 0.8;
    const float siftThresh = 40;
    struct _Point{
        int x;
        int y;
        _Point(){}
        _Point(int _x, int _y){
            x = _x;
            y = _y;
        }
    };
    struct _Point_pair{
        _Point left;
        _Point right;
        _Point_pair(_Point _left, _Point _right){
            left = _left;
            right = _right;
        }
    };
    
    
    // 通过knn 通过k-d tree
    vector<_Point_pair> MatchKeyPointByKDTree(vector<SiftDescriptor> left, vector<SiftDescriptor> right);
    // 通过计算欧拉记录的knn算法
    vector<_Point_pair> MatchKeyPoint(vector<SiftDescriptor> left, vector<SiftDescriptor> right);
    // RANSAC 算法去除outlier
    vector<_Point_pair> RANSAC(const vector<_Point_pair> & pairs);
    

}
namespace Homo{
    struct HomographyMatrix {
        float a, b, c, d, e, f, g, h;
        HomographyMatrix(float _a, float _b, float _c,
                         float _d, float _e, float _f, float _g, float _h) :
        a(_a), b(_b), c(_c), d(_d), e(_e), f(_f), g(_g), h(_h) {}
    };
    HomographyMatrix get_homography_matrix(const vector<Match::_Point_pair>& pair);
    float get_x_warped(float x, float y, HomographyMatrix H);
    float get_y_warped(float x, float y, HomographyMatrix H);
    HomographyMatrix get_Matrix(const vector<Match::_Point_pair> pairs);

}
namespace Match{
    vector<int> getIndexsOfInliner(const vector<_Point_pair> & pairs, Homo::HomographyMatrix H, set<int> seleted_index);
    
}
namespace Warp{
    float max_X_warped(const CImg<float> & src , Homo::HomographyMatrix H);
    float max_Y_warped(const CImg<float> & src , Homo::HomographyMatrix H);
    float min_X_warped(const CImg<float> & src , Homo::HomographyMatrix H);
    float min_Y_warped(const CImg<float> & src , Homo::HomographyMatrix H);
    
    void ShiftByOffset(const CImg<float> & src, CImg<float> & det, float offset_x, float offset_y);
    void homo_warping(const CImg<float> &src, CImg<float> &dst, Homo::HomographyMatrix H, float offset_x, float offset_y);
    void updateFeaturesByOffset(vector<SiftDescriptor> &feature,float offset_x, float offset_y);
    void updateFeaturesByHomography(vector<SiftDescriptor> &feature,
                                    Homo::HomographyMatrix H, float offset_x, float offset_y);
}
namespace Blend{
}


#endif /* utils_hpp */
