//
//  Stitching.cpp
//  CV-5-again
//
//  Created by Kluni on 2018/11/28.
//  Copyright © 2018 Kluni. All rights reserved.
//
#include "Stitching.hpp"
#include "utils.hpp"
const unsigned char color[] = {255, 0, 0};

CImg<unsigned char> multiband_blending(const CImg<unsigned char> &a, const CImg<unsigned char> &b) {
    /* Find overlapped area */
    int w = a.width(), h = a.height(); // a and b has the same size
    
    int sum_a_x = 0, sum_a_y = 0;
    int width_mid_a = 0; // width of image a's content (calculate in half of height)
    
    int sum_overlap_x = 0, sum_overlap_y = 0;
    int width_mid_overlap = 0;
    
    // only consider stitching image horizontally
    int mid_y = h / 2;
    int x = 0;
    while (a(x, mid_y) == 0) ++x; // avoid leading zero
    for (x; x < w; ++x) {
        if (a(x, mid_y) != 0) { // black
            sum_a_x += x;
            ++width_mid_a;
            if (b(x, mid_y) != 0) {
                sum_overlap_x += x;
                ++width_mid_overlap;
            }
        }
    }
    
    /* 0. Forming a Gaussian Pyramid */
    /* i. Start with the original image G0 */
    int max_len = w >= h ? w : h;
    int level_num = floor(log2(max_len));
    
    vector<CImg<float>> a_pyramid(level_num);
    vector<CImg<float>> b_pyramid(level_num);
    vector<CImg<float>> mask(level_num);
    
    a_pyramid[0] = a;
    b_pyramid[0] = b;
    mask[0] = CImg<float>(w, h, 1, 1, 0);
    assert(width_mid_a > 0);
    assert(width_mid_overlap > 0);
    float ratio = 1.0 * sum_a_x / width_mid_a;
    float overlap_ratio = 1.0 * sum_overlap_x / width_mid_overlap;
    // the x=overlap_ratio line should lie in the overlap area
    if (ratio < overlap_ratio) {
        for (int x = 0; x < overlap_ratio; ++x)
            for (int y = 0; y < h; ++y)
                mask[0](x, y) = 1;
    }
    else {
        for (int x = overlap_ratio + 1; x < w; ++x)
            for (int y = 0; y < h; ++y)
                mask[0](x, y) = 1;
    }
    
    
    /* ii. Perform a local Gaussian weighted averaging
     function in a neighborhood about each pixel,
     sampling so that the result is a reduced image
     of half the size in each dimension. */
    for (int i = 1; i < level_num; ++i) {
        int wp = a_pyramid[i - 1].width() / 2;
        int hp = a_pyramid[i - 1].height() / 2;
        int sp = a_pyramid[i - 1].spectrum();
        a_pyramid[i] = a_pyramid[i - 1].get_blur(2, true, true).
        get_resize(wp, hp, 1, sp, 3);
        b_pyramid[i] = b_pyramid[i - 1].get_blur(2, true, true).
        get_resize(wp, hp, 1, sp, 3);
        mask[i] = mask[i - 1].get_blur(2, true, true).
        get_resize(wp, hp, 1, sp, 3);
    } /* iii. Do this all the way up the pyramid Gl = REDUCE(Gl-1) */
    
    /* iiii. Each level l node will represent a weighted
     average of a subarray of level l. */
    
    
    
    /* 1. Compute Laplacian pyramid of images and mask */
    /* Making the Laplacians Li=Gi-expand(Gi+1)*/
    /*  subtract each level of the pyramid from the next lower one
     EXPAND:  interpolate new samples between those of
     a given image to make it big enough to subtract*/
    for (int i = 0; i < level_num - 1; ++i) {
        int wp = a_pyramid[i].width();
        int hp = a_pyramid[i].height();
        int sp = a_pyramid[i].spectrum();
        a_pyramid[i] -= a_pyramid[i + 1].get_resize(wp, hp, 1, sp, 3);
        b_pyramid[i] -= b_pyramid[i + 1].get_resize(wp, hp, 1, sp, 3);
    }
    
    
    /* 2. Create blended image at each level of pyramid */
    /* Forming the New Pyramid
     A third Laplacian pyramid LS is constructed by copying
     nodes from the left half of LA to the corresponding
     nodes of LS and nodes from the right half of LB to the
     right half of LS.
     Nodes along the center line are set equal to
     the average of corresponding LA and LB nodes */
    vector<CImg<float>> blend_pyramid(level_num);
    for (int i = 0; i < level_num; ++i) {
        blend_pyramid[i] = CImg<float>(a_pyramid[i].width(),
                                       a_pyramid[i].height(), 1, a_pyramid[i].spectrum(), 0);
        cimg_forXYC(blend_pyramid[i], x, y, c) {
            blend_pyramid[i](x, y, c) =
            a_pyramid[i](x, y, c) * mask[i](x, y) +
            b_pyramid[i](x, y, c) * (1.0 - mask[i](x, y));
        }
    }
    
    /* 3. Reconstruct complete image */
    /* Using the new Laplacian Pyramid
     Use the new Laplacian pyramid with the reverse of how it
     was created to create a Gaussian pyramid. Gi=Li+expand(Gi+1)
     The lowest level of the new Gaussian pyramid gives the final
     result. */
    // float: cannot be unsigned char(invalid) type!
    CImg<float> expand = blend_pyramid[level_num - 1];
    for (int i = level_num - 2; i >= 0; --i) {
        expand.resize(blend_pyramid[i].width(),
                      blend_pyramid[i].height(), 1, blend_pyramid[i].spectrum(), 3);
        cimg_forXYC(blend_pyramid[i], x, y, c) {
            expand(x, y, c) = blend_pyramid[i](x, y, c) + expand(x, y, c);
            if (expand(x, y, c) > 255) expand(x, y, c) = 255;
            else if (expand(x, y, c) < 0) expand(x, y, c) = 0;
        }
    }
    return expand;
}

void Stitching::stitch(string DirPath){
    vector<CImg<float>> image_vec;
    
    // 1. load images
    auto files = ReadFilesFromDir(DirPath);
    sort(files.begin(), files.end());
    
    for(int i = 0; i<files.size() ; i++){
        string filePath = DirPath + files[i];
        CImg<float> image(filePath.c_str());
        // preprocess image
        image_vec.push_back(preprocessing(image));
    }
    
    int mid = image_vec.size() / 2;  // 18 / 2 = 9
    CImg<float> stitched_img = image_vec[mid]; //  以最中间的图像初始化第一张图
    // 存储左右的关键点
    auto left_points = Sift::compute_sift(image_vec[mid]);
    auto right_points = left_points;
    
    for (int i = 1; i<image_vec.size(); i++) {
        int to_stitch_index ;
        vector<SiftDescriptor> mid_points;
        if (i % 2 == 1){  // 奇数往右偶数往左
            to_stitch_index = mid + (i+1)/2;
            if(to_stitch_index == image_vec.size())continue;
            mid_points = right_points;
        }
        else{
            to_stitch_index = mid - i/2;
            if (to_stitch_index < 0) continue;
            mid_points = left_points;
        }
        
        CImg<float> adjacent_img(image_vec[to_stitch_index]);
        // get the key points of the right image
        auto adj_points = Sift::compute_sift(adjacent_img);
        
        // match them
//        auto left_2_right = Match::MatchKeyPointByKDTree(mid_points, adj_points);
//        auto right_2_left = Match::MatchKeyPointByKDTree(adj_points, mid_points);
        
        auto left_2_right = Match::MatchKeyPoint(mid_points, adj_points);
        auto right_2_left = Match::MatchKeyPoint(adj_points, mid_points);
        
        if (left_2_right.size() > right_2_left.size()) {
            right_2_left.clear();
            for (auto i : left_2_right) {
                right_2_left.push_back(Match::_Point_pair(i.right, i.left));
            }
        }else{
            left_2_right.clear();
            for (auto i : right_2_left) {
                left_2_right.push_back(Match::_Point_pair(i.right, i.left));
            }
        }
        
        // apple RANSAC
        auto pair_after_ransac = Match::RANSAC(left_2_right);
        auto pair_after_ransac_inv = Match::RANSAC(right_2_left);
        
        // make homography by pairs
        
        Homo::HomographyMatrix H = Homo::get_Matrix(pair_after_ransac);
        Homo::HomographyMatrix H_inv = Homo::get_Matrix(pair_after_ransac_inv);
        
        
        // Display the pairs
        CImg<float> BIG(stitched_img.width() + adjacent_img.width() , stitched_img.height(), 1, 3, 0);
        cimg_forXYZC(adjacent_img, x, y, z, c) BIG(x, y, z, c) = adjacent_img(x, y, z, c);
        cimg_forXYZC(stitched_img, x, y, z, c) BIG(x + adjacent_img.width(), y, z, c) = stitched_img(x, y, z, c);
        for (auto i : pair_after_ransac) {
            BIG.draw_line(i.right.x, i.right.y, i.left.x + adjacent_img.width(), i.left.y, color);
        }
        BIG.display("BIG");
        // Calculat the size after stitching
        float min_x = Warp::min_X_warped(adjacent_img, H_inv);
        min_x = (min_x < 0) ? min_x : 0;
        float min_y = Warp::min_Y_warped(adjacent_img, H_inv);
        min_y = (min_y < 0) ? min_y : 0;
        float max_x = Warp::max_X_warped(adjacent_img, H_inv);
        max_x = (max_x >= stitched_img.width()) ? max_x : stitched_img.width();
        float max_y = Warp::max_Y_warped(adjacent_img, H_inv);
        max_y = (max_y > stitched_img.height()) ? max_y : stitched_img.height();
        
        
        int out_w = ceil(max_x - min_x);
        int out_h = ceil(max_y - min_y);
        
        CImg<float> last_stitch(out_w, out_h, 1, adjacent_img.spectrum(), 0);
        CImg<float> next_stitch(out_w, out_h, 1, adjacent_img.spectrum(), 0);

        Warp::ShiftByOffset(stitched_img, last_stitch, min_x, min_y);
        Warp::homo_warping(adjacent_img, next_stitch, H, min_x, min_y);
        
        last_stitch.display("last stitch");
        next_stitch.display("nextt stitch");
        
        // We can Undo ~~
        CImgDisplay dsp;
        auto temp = multiband_blending(last_stitch, next_stitch);
        dsp.display(temp);
        bool undo = false;
        while (!dsp.is_closed()) if (dsp.button()&1) undo = true;
        if (undo) {
            i--;
            continue;
        }
        // update the key points
//        Warp::updateFeaturesByOffset(mid_points, min_x, min_y);
        Warp::updateFeaturesByHomography(adj_points, H_inv, min_x, min_y);
        
        stitched_img = temp;
        if (i%2 == 1) {
            right_points = adj_points;
            Warp::updateFeaturesByOffset(left_points, min_x, min_y);

//            left_points = mid_points;
        }
        else {
            left_points = adj_points;
            Warp::updateFeaturesByOffset(right_points, min_x, min_y);

//            right_points = mid_points;
        }
    }
}

CImg<float> Stitching::preprocessing(CImg<float> image){
    return sphericalProjection(image.resize(image._width*scale,image._height*scale));
}
CImg<float> Stitching::RGB2Gary(CImg<float> image){
    if(image.spectrum() == 1) return image;
    CImg<unsigned char> temp(image._width, image._height,1);
    cimg_forXY(temp, x, y){
        int b = image(x, y, 0);
        int g = image(x, y, 1);
        int r = image(x, y, 2);
        double n = (r * 0.2126 + g * 0.7152 + b * 0.0722);
        temp(x,y) = n;
    }
    return temp;
}


