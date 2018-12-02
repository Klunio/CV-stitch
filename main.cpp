//
//  main.cpp
//  CV-5-again
//
//  Created by Kluni on 2018/11/28.
//  Copyright © 2018 Kluni. All rights reserved.
//
#include <iostream>
#include <stdio.h>
#include <dirent.h>
#include <vector>
#include "Stitching.hpp"
using namespace std;
void test(){
//    char inputPath[] = "/Users/guyunquan/Desktop/计算机视觉/作业/作业4/作业4/TEST-ImageData(2)/";
    char inputPath[] = "/Users/guyunquan/Desktop/test/";
    char outputPath[] = "/Users/guyunquan/ComputerVision/Ex6/result/";
    
    Stitching a;
    
    a.stitch(inputPath);
}


int main(){
    cimg::imagemagick_path("/Users/guyunquan/ImageMagick-7.0.8/bin/magick");
    
    test();
    
    
    return 0;
}
