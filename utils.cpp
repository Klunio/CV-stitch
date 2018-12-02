//
//  utils.cpp
//  CV-5-again
//
//  Created by Kluni on 2018/12/1.
//  Copyright © 2018 Kluni. All rights reserved.
//

#include "utils.hpp"
//#include "kd-tree/kdtree.c"
//#include "kd-tree/generic.c"
//#include "kd-tree/host.c"
//#include "kd-tree/mathop.c"
using namespace cimg_library;

vector<string> ReadFilesFromDir(string DirPath){
    DIR *dir;
    vector<string> files;
    struct dirent *ptr;
    if ((dir = opendir(DirPath.c_str())) == NULL) {
        perror("fail to open directory!");
        exit(1);
    }
    while ((ptr = readdir(dir)) != NULL) {
        if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0) continue;
        files.push_back(ptr->d_name);
    }
    closedir(dir);
    return files;
}
CImg<float> sphericalProjection(const CImg<float> &src){
    CImg<float> res(src.width(), src.height(), 1, src.spectrum(), 0);

    int larger = src._height * src._height + src._width * src._width;
    
    float r = (sqrt(larger) / 2) / tan(sphericalAngle * cimg::PI / 180.0);
    
    cimg_forXY(res, x, y){
        float dst_x = x - res.width() / 2;
        float dst_y = y - res.height() / 2;
        
        float k = r / sqrt(r * r - dst_x * dst_x - dst_y* dst_y);
        
        float src_x = dst_x * k;
        float src_y = dst_y * k;
        
        if (src_x + src.width() / 2 >= 0 && src_x + src.width() / 2 < src.width()
            && src_y + src.height() / 2 >= 0 && src_y + src.height() / 2 < src.height()) {
            for (int k = 0; k < res.spectrum(); k++) {
                res(x, y, k) = bilinear_interpolation(src, src_x + src.width() / 2, src_y + src.height() / 2,  k);
            }
        }
    }
    return res;
}
float bilinear_interpolation(const CImg<float>& image, float x, float y, int channel) {

    assert(x >= 0 && x < image.width());
    assert(y >= 0 && y < image.height());
    assert(channel <= image.spectrum());
    
    int x_pos = floor(x);
    float x_u = x - x_pos;
    int xb = (x_pos < image.width() - 1) ? x_pos + 1 : x_pos;
    
    int y_pos = floor(y);
    float y_v = y - y_pos;
    int yb = (y_pos < image.height() - 1) ? y_pos + 1 : y_pos;
    
    float P1 = image(x_pos, y_pos, channel) * (1 - x_u) + image(xb, y_pos, channel) * x_u;
    float P2 = image(x_pos, yb, channel) * (1 - x_u) + image(xb, yb, channel) * x_u;
    
    return P1 * (1 - y_v) + P2 * y_v;
}
namespace Match {
    vector<_Point_pair> MatchKeyPoint(vector<SiftDescriptor> left, vector<SiftDescriptor> right){
        vector<_Point_pair> vec;
        for (auto i : left) {
            float nn1 = 100000 , nn2 = 100000;
            SiftDescriptor bestindex;
            for (auto j : right) {
                float sum = 0;
                for (int k = 0; k < 128; k++) {
                    sum += pow(i.descriptor[k] - j.descriptor[k], 2.0);
                }
                sum = sqrt(sum);
                if (sum < nn1) {
                    nn1 = sum;
                    bestindex = j;
                }
                else if(sum < nn2) nn2 = sum;
            }
            
            if (nn1 / nn2 < Match::ratioThresh && nn1 < Match::siftThresh) {
                vec.push_back(_Point_pair(_Point(i.col, i.row), _Point(bestindex.col, bestindex.row)));
            }
        }
        return vec;
    }
    vector<_Point_pair> RANSAC(const vector<_Point_pair> &pairs){
        assert(pairs.size() >= NUM_OF_PAIR);
        srand(time(0));
        
        int iterations = ceil(log(1 - CONFIDENCE) / log(1 - pow(INLINER_RATIO, NUM_OF_PAIR)));
        vector<int> max_inliner_indexs;
        
        while (iterations--) {
            vector<_Point_pair> random_sample;
            set<int> seleted_index;
            
            for (int i = 0; i < NUM_OF_PAIR; i++) {
                
                int index = rand() % (pairs.size() - 1);
                while (seleted_index.find(index) != seleted_index.end()) {
                    index = rand() % (pairs.size() - 1);
                }
                seleted_index.insert(index);
                
                random_sample.push_back(pairs[index]);
            }
            
            Homo::HomographyMatrix H = Homo::get_homography_matrix(random_sample);
            
            vector<int> cur_inliner_index = getIndexsOfInliner(pairs, H, seleted_index);
            if (cur_inliner_index.size() > max_inliner_indexs.size())
                max_inliner_indexs = cur_inliner_index;
        }
        vector<_Point_pair> res;
        for(auto i : max_inliner_indexs) res.push_back(pairs[i]);
        
        return res;
    }
    vector<int> getIndexsOfInliner(const vector<_Point_pair> & pairs, Homo::HomographyMatrix H, set<int> seleted_index){
        vector<int> inliner_indexs;
        for (int i = 0; i < pairs.size(); i++) {
            if (seleted_index.find(i) != seleted_index.end()) continue;
            
            float real_x = pairs[i].right.x;
            float real_y = pairs[i].right.y;
            
            float x = get_x_warped(pairs[i].left.x, pairs[i].left.y, H);
            float y = get_y_warped(pairs[i].left.x, pairs[i].left.y, H);
            
            float distance = sqrt((x - real_x) * (x - real_x) + (y - real_y) * (y - real_y));
            if (distance < RANSAC_THRESHOLD) inliner_indexs.push_back(i);
        }
        return inliner_indexs;
    }
    static double dist_sq( double *a1, double *a2, int dims ) {
        double dist_sq = 0, diff;
        while( --dims >= 0 ) {
            diff = (a1[dims] - a2[dims]);
            dist_sq += diff*diff;
        }
        return dist_sq;
    }
    vector<_Point_pair> MatchKeyPointByKDTree(vector<SiftDescriptor> left, vector<SiftDescriptor> right){
        vector<_Point_pair> res;
        /* create a k-d tree for 128-dimensional points*/
        
    
        auto ptree = kd_create(128);
        int* pin;
        int* index = (int*)malloc(left.size());
        double pos[128];
        printf("Seeding %d entries of %d dimensions...\n", (int)left.size(), 128);

        double data[128];
        for(int i = 0 ; i < left.size() ; i++){
            index[i] = i;
            for (int j = 0; j<128; j++) data[j] = left[i].descriptor[j];
            assert( 0 == kd_insert(ptree, data, &index[i]));
        }

        struct kdres *presults;

        for (auto it = right.begin(); it != right.end(); it++) {
            float *pt = new float[128];
            for (int i = 0; i<128; i++) pt[i] = (it->descriptor)[i];

            // 找这个pt最近的

            presults = kd_nearest_range(ptree, (double*)pt, siftThresh);
            printf( "found %d results:\n", kd_res_size(presults) );

            while (!kd_res_end(presults)) {
                int nn1,nn2;
                double m1= 10000,m2 = 10000;
                pin = (int*)kd_res_item(presults, pos);
                double dist = sqrt(dist_sq((double*)pt, pos, 128));

                if (dist < m1) {
                    m1 = dist;
                    nn1 = *pin;
                }else if(dist < m2) m2 = dist;

                if (nn1 / nn2 < ratioThresh) {
                    _Point l(left[nn1].col, left[nn1].row);
                    _Point r(it->col, it->row);
                    res.push_back(_Point_pair(l, r));
                }

                kd_res_next(presults);

            }
            delete[] pt;
        }

        free(index);
        kd_res_free(presults);
        kd_free(ptree);
        return res;
    }

}


namespace Homo {
    HomographyMatrix get_homography_matrix(const vector<Match::_Point_pair>& pair){
        assert(pair.size() == 4);
        
        float u0 = pair[0].left.x, v0 = pair[0].left.y;
        float u1 = pair[1].left.x, v1 = pair[1].left.y;
        float u2 = pair[2].left.x, v2 = pair[2].left.y;
        float u3 = pair[3].left.x, v3 = pair[3].left.y;
        
        float x0 = pair[0].right.x, y0 = pair[0].right.y;
        float x1 = pair[1].right.x, y1 = pair[1].right.y;
        float x2 = pair[2].right.x, y2 = pair[2].right.y;
        float x3 = pair[3].right.x, y3 = pair[3].right.y;
        
        float a, b, c, d, e, f, g, h;
        
        a = -(u0*v0*v1*x2 - u0*v0*v2*x1 - u0*v0*v1*x3 + u0*v0*v3*x1 - u1*v0*v1*x2 + u1*v1*v2*x0 + u0*v0*v2*x3 - u0*v0*v3*x2 + u1*v0*v1*x3 - u1*v1*v3*x0 + u2*v0*v2*x1 - u2*v1*v2*x0
              - u1*v1*v2*x3 + u1*v1*v3*x2 - u2*v0*v2*x3 + u2*v2*v3*x0 - u3*v0*v3*x1 + u3*v1*v3*x0 + u2*v1*v2*x3 - u2*v2*v3*x1 + u3*v0*v3*x2 - u3*v2*v3*x0 - u3*v1*v3*x2 + u3*v2*v3*x1)
        / (u0*u1*v0*v2 - u0*u2*v0*v1 - u0*u1*v0*v3 - u0*u1*v1*v2 + u0*u3*v0*v1 + u1*u2*v0*v1 + u0*u1*v1*v3 + u0*u2*v0*v3 + u0*u2*v1*v2 - u0*u3*v0*v2 - u1*u2*v0*v2 - u1*u3*v0*v1
           - u0*u2*v2*v3 - u0*u3*v1*v3 - u1*u2*v1*v3 + u1*u3*v0*v3 + u1*u3*v1*v2 + u2*u3*v0*v2 + u0*u3*v2*v3 + u1*u2*v2*v3 - u2*u3*v0*v3 - u2*u3*v1*v2 - u1*u3*v2*v3 + u2*u3*v1*v3);
        
        b = (u0*u1*v0*x2 - u0*u2*v0*x1 - u0*u1*v0*x3 - u0*u1*v1*x2 + u0*u3*v0*x1 + u1*u2*v1*x0 + u0*u1*v1*x3 + u0*u2*v0*x3 + u0*u2*v2*x1 - u0*u3*v0*x2 - u1*u2*v2*x0 - u1*u3*v1*x0
             - u0*u2*v2*x3 - u0*u3*v3*x1 - u1*u2*v1*x3 + u1*u3*v1*x2 + u1*u3*v3*x0 + u2*u3*v2*x0 + u0*u3*v3*x2 + u1*u2*v2*x3 - u2*u3*v2*x1 - u2*u3*v3*x0 - u1*u3*v3*x2 + u2*u3*v3*x1)
        / (u0*u1*v0*v2 - u0*u2*v0*v1 - u0*u1*v0*v3 - u0*u1*v1*v2 + u0*u3*v0*v1 + u1*u2*v0*v1 + u0*u1*v1*v3 + u0*u2*v0*v3 + u0*u2*v1*v2 - u0*u3*v0*v2 - u1*u2*v0*v2 - u1*u3*v0*v1
           - u0*u2*v2*v3 - u0*u3*v1*v3 - u1*u2*v1*v3 + u1*u3*v0*v3 + u1*u3*v1*v2 + u2*u3*v0*v2 + u0*u3*v2*v3 + u1*u2*v2*v3 - u2*u3*v0*v3 - u2*u3*v1*v2 - u1*u3*v2*v3 + u2*u3*v1*v3);
        
        c = (u0*v1*x2 - u0*v2*x1 - u1*v0*x2 + u1*v2*x0 + u2*v0*x1 - u2*v1*x0 - u0*v1*x3 + u0*v3*x1 + u1*v0*x3 - u1*v3*x0 - u3*v0*x1 + u3*v1*x0
             + u0*v2*x3 - u0*v3*x2 - u2*v0*x3 + u2*v3*x0 + u3*v0*x2 - u3*v2*x0 - u1*v2*x3 + u1*v3*x2 + u2*v1*x3 - u2*v3*x1 - u3*v1*x2 + u3*v2*x1)
        / (u0*u1*v0*v2 - u0*u2*v0*v1 - u0*u1*v0*v3 - u0*u1*v1*v2 + u0*u3*v0*v1 + u1*u2*v0*v1 + u0*u1*v1*v3 + u0*u2*v0*v3 + u0*u2*v1*v2 - u0*u3*v0*v2 - u1*u2*v0*v2 - u1*u3*v0*v1
           - u0*u2*v2*v3 - u0*u3*v1*v3 - u1*u2*v1*v3 + u1*u3*v0*v3 + u1*u3*v1*v2 + u2*u3*v0*v2 + u0*u3*v2*v3 + u1*u2*v2*v3 - u2*u3*v0*v3 - u2*u3*v1*v2 - u1*u3*v2*v3 + u2*u3*v1*v3);
        
        d = (u0*u1*v0*v2*x3 - u0*u1*v0*v3*x2 - u0*u2*v0*v1*x3 + u0*u2*v0*v3*x1 + u0*u3*v0*v1*x2 - u0*u3*v0*v2*x1 - u0*u1*v1*v2*x3 + u0*u1*v1*v3*x2 + u1*u2*v0*v1*x3 - u1*u2*v1*v3*x0 - u1*u3*v0*v1*x2 + u1*u3*v1*v2*x0
             + u0*u2*v1*v2*x3 - u0*u2*v2*v3*x1 - u1*u2*v0*v2*x3 + u1*u2*v2*v3*x0 + u2*u3*v0*v2*x1 - u2*u3*v1*v2*x0 - u0*u3*v1*v3*x2 + u0*u3*v2*v3*x1 + u1*u3*v0*v3*x2 - u1*u3*v2*v3*x0 - u2*u3*v0*v3*x1 + u2*u3*v1*v3*x0)
        / (u0*u1*v0*v2 - u0*u2*v0*v1 - u0*u1*v0*v3 - u0*u1*v1*v2 + u0*u3*v0*v1 + u1*u2*v0*v1 + u0*u1*v1*v3 + u0*u2*v0*v3 + u0*u2*v1*v2 - u0*u3*v0*v2 - u1*u2*v0*v2 - u1*u3*v0*v1
           - u0*u2*v2*v3 - u0*u3*v1*v3 - u1*u2*v1*v3 + u1*u3*v0*v3 + u1*u3*v1*v2 + u2*u3*v0*v2 + u0*u3*v2*v3 + u1*u2*v2*v3 - u2*u3*v0*v3 - u2*u3*v1*v2 - u1*u3*v2*v3 + u2*u3*v1*v3);
        
        e = -(u0*v0*v1*y2 - u0*v0*v2*y1 - u0*v0*v1*y3 + u0*v0*v3*y1 - u1*v0*v1*y2 + u1*v1*v2*y0 + u0*v0*v2*y3 - u0*v0*v3*y2 + u1*v0*v1*y3 - u1*v1*v3*y0 + u2*v0*v2*y1 - u2*v1*v2*y0
              - u1*v1*v2*y3 + u1*v1*v3*y2 - u2*v0*v2*y3 + u2*v2*v3*y0 - u3*v0*v3*y1 + u3*v1*v3*y0 + u2*v1*v2*y3 - u2*v2*v3*y1 + u3*v0*v3*y2 - u3*v2*v3*y0 - u3*v1*v3*y2 + u3*v2*v3*y1)
        / (u0*u1*v0*v2 - u0*u2*v0*v1 - u0*u1*v0*v3 - u0*u1*v1*v2 + u0*u3*v0*v1 + u1*u2*v0*v1 + u0*u1*v1*v3 + u0*u2*v0*v3 + u0*u2*v1*v2 - u0*u3*v0*v2 - u1*u2*v0*v2 - u1*u3*v0*v1
           - u0*u2*v2*v3 - u0*u3*v1*v3 - u1*u2*v1*v3 + u1*u3*v0*v3 + u1*u3*v1*v2 + u2*u3*v0*v2 + u0*u3*v2*v3 + u1*u2*v2*v3 - u2*u3*v0*v3 - u2*u3*v1*v2 - u1*u3*v2*v3 + u2*u3*v1*v3);
        
        f = (u0*u1*v0*y2 - u0*u2*v0*y1 - u0*u1*v0*y3 - u0*u1*v1*y2 + u0*u3*v0*y1 + u1*u2*v1*y0 + u0*u1*v1*y3 + u0*u2*v0*y3 + u0*u2*v2*y1 - u0*u3*v0*y2 - u1*u2*v2*y0 - u1*u3*v1*y0
             - u0*u2*v2*y3 - u0*u3*v3*y1 - u1*u2*v1*y3 + u1*u3*v1*y2 + u1*u3*v3*y0 + u2*u3*v2*y0 + u0*u3*v3*y2 + u1*u2*v2*y3 - u2*u3*v2*y1 - u2*u3*v3*y0 - u1*u3*v3*y2 + u2*u3*v3*y1)
        / (u0*u1*v0*v2 - u0*u2*v0*v1 - u0*u1*v0*v3 - u0*u1*v1*v2 + u0*u3*v0*v1 + u1*u2*v0*v1 + u0*u1*v1*v3 + u0*u2*v0*v3 + u0*u2*v1*v2 - u0*u3*v0*v2 - u1*u2*v0*v2 - u1*u3*v0*v1
           - u0*u2*v2*v3 - u0*u3*v1*v3 - u1*u2*v1*v3 + u1*u3*v0*v3 + u1*u3*v1*v2 + u2*u3*v0*v2 + u0*u3*v2*v3 + u1*u2*v2*v3 - u2*u3*v0*v3 - u2*u3*v1*v2 - u1*u3*v2*v3 + u2*u3*v1*v3);
        
        g = (u0*v1*y2 - u0*v2*y1 - u1*v0*y2 + u1*v2*y0 + u2*v0*y1 - u2*v1*y0 - u0*v1*y3 + u0*v3*y1 + u1*v0*y3 - u1*v3*y0 - u3*v0*y1 + u3*v1*y0
             + u0*v2*y3 - u0*v3*y2 - u2*v0*y3 + u2*v3*y0 + u3*v0*y2 - u3*v2*y0 - u1*v2*y3 + u1*v3*y2 + u2*v1*y3 - u2*v3*y1 - u3*v1*y2 + u3*v2*y1)
        / (u0*u1*v0*v2 - u0*u2*v0*v1 - u0*u1*v0*v3 - u0*u1*v1*v2 + u0*u3*v0*v1 + u1*u2*v0*v1 + u0*u1*v1*v3 + u0*u2*v0*v3 + u0*u2*v1*v2 - u0*u3*v0*v2 - u1*u2*v0*v2 - u1*u3*v0*v1
           - u0*u2*v2*v3 - u0*u3*v1*v3 - u1*u2*v1*v3 + u1*u3*v0*v3 + u1*u3*v1*v2 + u2*u3*v0*v2 + u0*u3*v2*v3 + u1*u2*v2*v3 - u2*u3*v0*v3 - u2*u3*v1*v2 - u1*u3*v2*v3 + u2*u3*v1*v3);
        
        h = (u0*u1*v0*v2*y3 - u0*u1*v0*v3*y2 - u0*u2*v0*v1*y3 + u0*u2*v0*v3*y1 + u0*u3*v0*v1*y2 - u0*u3*v0*v2*y1 - u0*u1*v1*v2*y3 + u0*u1*v1*v3*y2 + u1*u2*v0*v1*y3 - u1*u2*v1*v3*y0 - u1*u3*v0*v1*y2 + u1*u3*v1*v2*y0
             + u0*u2*v1*v2*y3 - u0*u2*v2*v3*y1 - u1*u2*v0*v2*y3 + u1*u2*v2*v3*y0 + u2*u3*v0*v2*y1 - u2*u3*v1*v2*y0 - u0*u3*v1*v3*y2 + u0*u3*v2*v3*y1 + u1*u3*v0*v3*y2 - u1*u3*v2*v3*y0 - u2*u3*v0*v3*y1 + u2*u3*v1*v3*y0)
        / (u0*u1*v0*v2 - u0*u2*v0*v1 - u0*u1*v0*v3 - u0*u1*v1*v2 + u0*u3*v0*v1 + u1*u2*v0*v1 + u0*u1*v1*v3 + u0*u2*v0*v3 + u0*u2*v1*v2 - u0*u3*v0*v2 - u1*u2*v0*v2 - u1*u3*v0*v1
           - u0*u2*v2*v3 - u0*u3*v1*v3 - u1*u2*v1*v3 + u1*u3*v0*v3 + u1*u3*v1*v2 + u2*u3*v0*v2 + u0*u3*v2*v3 + u1*u2*v2*v3 - u2*u3*v0*v3 - u2*u3*v1*v2 - u1*u3*v2*v3 + u2*u3*v1*v3);
        
        return HomographyMatrix(a, b, c, d, e, f, g, h);
    }
    float get_x_warped(float x, float y, HomographyMatrix H){
        return H.a * x + H.b * y + H.c * x * y + H.d;
    }
    float get_y_warped(float x, float y, HomographyMatrix H){
        return H.e * x + H.f * y + H.g * x * y + H.h;
    }
    HomographyMatrix get_Matrix(const vector<Match::_Point_pair> pairs){
        int size = (int)pairs.size();
        CImg<double> A(4, size, 1, 1, 0);
        CImg<double> b(1, size, 1, 1, 0);
        
        for (int i = 0; i<size; i++) {
            A(0, i) = pairs[i].left.x;
            A(1, i) = pairs[i].left.y;
            A(2, i) = pairs[i].left.x * pairs[i].left.y;
            A(3, i) = 1;
            
            b(0, i) = pairs[i].right.x;
        }
        CImg<double> x1 = b.get_solve(A);
        
        for (int i = 0; i< size; i++) {
            b(0, i) = pairs[i].right.y;
        }
        
        CImg<double> x2 = b.get_solve(A);
        
        return HomographyMatrix(x1(0, 0), x1(0, 1), x1(0, 2), x1(0, 3), x2(0, 0), x2(0, 1), x2(0, 2), x2(0, 3));
    }

}
namespace Warp{
    using namespace Homo;

    float max_X_warped(const CImg<float> & src , Homo::HomographyMatrix H){
        int w = src.width();
        int h = src.height();
        float max = cimg::max(get_x_warped(0, 0, H), get_x_warped(w - 1, 0, H),
                              get_x_warped(0, h -1, H), get_x_warped(w - 1, h - 1 , H));
        return max;
    }
    float max_Y_warped(const CImg<float> & src , Homo::HomographyMatrix H){
        int w = src.width();
        int h = src.height();
        float max = cimg::max(get_y_warped(0, 0, H), get_y_warped(w - 1, 0, H),
                              get_y_warped(0, h -1, H), get_y_warped(w - 1, h - 1 , H));
        return max;
    }
    float min_X_warped(const CImg<float> & src , Homo::HomographyMatrix H){
        int w = src.width();
        int h = src.height();
        float min = cimg::min(get_x_warped(0, 0, H), get_x_warped(w - 1, 0, H),
                              get_x_warped(0, h -1, H), get_x_warped(w - 1, h - 1 , H));
        return min;
    }
    float min_Y_warped(const CImg<float> & src , Homo::HomographyMatrix H){
        int w = src.width();
        int h = src.height();
        float min = cimg::min(get_y_warped(0, 0, H), get_y_warped(w - 1, 0, H),
                              get_y_warped(0, h -1, H), get_y_warped(w - 1, h - 1 , H));
        return min;
    }
    void ShiftByOffset(const CImg<float> & src, CImg<float> & dst, float offset_x, float offset_y){
        cimg_forXY(dst, x, y){
            int x_ = x + offset_x;
            int y_ = y + offset_y;
            if(x_ >= 0 && x_ < src.width() && y_ >=0 && y_ < src.height())
                cimg_forC(dst, c) dst(x, y, c) = src(x_, y_, c);
        }
    }
    void homo_warping(const CImg<float> &src, CImg<float> &dst, HomographyMatrix H, float offset_x, float offset_y){
        cimg_forXY(dst, x, y) {
            float warped_x = get_x_warped(x + offset_x, y + offset_y, H);
            float warped_y = get_y_warped(x + offset_x, y + offset_y, H);
            if (warped_x >= 0 && warped_x < src.width() &&
                warped_y >= 0 && warped_y < src.height()) {
                cimg_forC(dst, c) {
                    dst(x, y, c) = src(floor(warped_x), floor(warped_y), c);
                }
            }
        }
    }
    void updateFeaturesByOffset(vector<SiftDescriptor> &feature, float offset_x, float offset_y) {
        vector<SiftDescriptor>::iterator it = feature.begin();
        for (it; it != feature.end(); ++it) {
            it->col -= offset_x; // coordinate
            it->row -= offset_y;
        }
    }
    void updateFeaturesByHomography(vector<SiftDescriptor> &feature,
                                    HomographyMatrix H, float offset_x, float offset_y) {
        vector<SiftDescriptor>::iterator it = feature.begin();
        for (it; it != feature.end(); ++it) {
            float px = it->col;// coordinate
            float py = it->row;
            it->col = get_x_warped(px, py, H) - offset_x;
            it->row = get_y_warped(px, py, H) - offset_y;
        }
    }
    
    
}
