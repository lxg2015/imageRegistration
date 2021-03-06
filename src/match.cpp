#include "match.h"
#include "gms.h"
#include "tool.h"
#include <iostream>
#include <map>

using namespace cv;
using namespace std;

enum FEATURE {surf, orb};

Match::Match(){
    method = CV_LMEDS; //CV_RANSAC; 
    outlier_thresh = 2;
    max_iter = 600;
    conf = 0.96;
    use_gms = true;
    use_scale = true;
    distance_thresh = 0.2; 

    FEATURE flg = orb;
    map<string, float> mv;
    
    load_config("./config.yaml", mv);
    if(mv["feature"] == 0){
        flg = surf;
    }

    if(flg == surf){
        feature = xfeatures2d::SURF::create(400); // 海塞矩阵阈值，越大特征越明显
        matcher = DescriptorMatcher::create("FlannBased");
        distance_thresh = 0.2; // 0 - 1
        cout << "\nfeature: surf\n";
    } else if(flg == orb){
        feature = ORB::create(100000); // mask 会影响特征提取个数
        matcher = DescriptorMatcher::create("BruteForce-Hamming");
        Ptr<ORB> orb = feature.dynamicCast<ORB>(); 
        orb->setFastThreshold(0);
        distance_thresh = 60; // > 1
        cout << "\nfeature: orb\n";
    } else {
        cout << "\nmust choose feature which to use\n";
    }
    cout << "\n";
}

void Match::get_feat_desc(Mat &img, Mat &mask, Mat &label, vector<Mask_Points> &feats, vector<Mat> &descs){
    vector<KeyPoint> points;
    Mat descrip;
    feature->detectAndCompute(img, mask, points, descrip);
    cout << "get feature:" << points.size() << endl;

    vector<Mask_Descs> des;
    feats.resize(num_area);
    descs.resize(num_area);
    des.resize(num_area);

    for(int i = 0; i < points.size(); ++i){
        Point2f &p = points[i].pt;
        short int flg = label.at<short int>(int(p.y), int(p.x));
        feats[flg].push_back(points[i]);
        des[flg].push_back(descrip.row(i));
    }
    cout << "reorgnize feature\n";
    for(int i = 0; i < num_area; ++i){
        Mat fea;
        vconcat(des[i], fea);
        descs[i] = fea;
    }
}

void Match::load_tpl(string name){
    tpl_img = imread(name);
    assert(!tpl_img.empty()) ;

    int n = name.length();
    tpl_mask = imread(name.substr(0, n-4) + "_mask.png", IMREAD_GRAYSCALE);
    if(tpl_mask.empty()){
        cout << "have not find _mask.png" << endl;
        tpl_mask = Mat::ones(tpl_img.size(), CV_8UC1);
    }
    assert(tpl_img.size() == tpl_mask.size());
    num_area = connectedComponents(tpl_mask, tpl_label, 8, CV_16U);
    cout << "num_area:" << num_area << endl;
    tpl_feats.clear();
    tpl_descs.clear();
    get_feat_desc(tpl_img, tpl_mask, tpl_label, tpl_feats, tpl_descs);

    // show
    cout << "load tpl done\n\n";
    Mat dst;
    cvtColor(tpl_mask, dst, CV_GRAY2BGR);
    addWeighted(tpl_img, 0.6, dst, 0.4, 0.3, tpl_img);
    draw_point(tpl_img, tpl_feats);
    imwrite("../img/template.png", tpl_img);
}

void Match::load_src(string name){
    src_img = imread(name);
    assert(!src_img.empty());
    src_feats.clear();
    src_descs.clear();
    Mat mask = Mat::ones(src_img.size(), CV_8U);
    get_feat_desc(src_img, tpl_mask, tpl_label, src_feats, src_descs);
    
    // show
    cout << "load src done\n\n";   
    Mat dst;
    cvtColor(tpl_mask, dst, CV_GRAY2BGR);
    addWeighted(src_img, 0.6, dst, 0.4, 0.3, src_img); 
    draw_point(src_img, src_feats);
    imwrite("../img/src.png", src_img);
}

void Match::compute_simi(){
    vector<Point2f> tpl, src;

    for(int i = 0; i < num_area; ++i){
        if(tpl_descs[i].empty() | src_descs[i].empty()) continue;
        matcher->match(src_descs[i], tpl_descs[i], match_points);
        sort(match_points.begin(), match_points.end());
        Mask_Points &tmp = tpl_feats[i];
        Mask_Points &smp = src_feats[i];
        
        int max_match = min(tmp.size(), smp.size());
        cout << i << " tpl:" << tpl_descs[i].size() << " src:" << src_descs[i].size() << " " << max_match << endl;

        // gms, 
        // todo, do gms in each area
        vector<bool> inlier;
        GMSMatcher gms = GMSMatcher(smp, src_img.size(), tmp, tpl_img.size(), match_points, 6);
        if(use_gms){
            gms.getInlierMask(inlier, false, false);
        }

        for(int j = 0; j < max_match; ++j){
            if (match_points[j].distance > distance_thresh) continue;
            if (use_gms && !inlier[j]) continue;
            tpl.push_back(tmp[match_points[j].trainIdx].pt);
            src.push_back(smp[match_points[j].queryIdx].pt);
        }
    }

    Mat inlier;
    assert(tpl.size() > 0);
    transform = estimateAffinePartial2D(tpl, src, inlier, method, outlier_thresh, max_iter, conf);
    cout << "\n" << transform << endl;
    
    // scale
    if(use_scale){
        double sn = transform.at<double>(0, 0);
        double cs = transform.at<double>(0, 1);
        double scale = sqrt(sn * sn + cs * cs);
        cout << scale << endl;
        transform.at<double>(0, 0) /= scale;
        transform.at<double>(0, 1) /= scale;
        transform.at<double>(1, 0) /= scale;
        transform.at<double>(1, 1) /= scale;
    }

    // show
    cout << "\nmatch size: " << src.size() << endl;
    Mat dst;
    cout << tpl_img.size() << " " << src_img.size() << endl;
    hconcat(src_img, tpl_img, dst);
    for(int i = 0; i < tpl.size(); ++i){
        int x1 = int(src[i].x);
        int y1 = int(src[i].y);
        int x2 = int(tpl[i].x) + tpl_img.cols;
        int y2 = int(tpl[i].y);
        char flg = inlier.at<char>(i, 0);
        Scalar color = flg == 1 ? Scalar(0, 128, 0) : Scalar(0, 0, 128);
        line(dst, {x1, y1}, {x2, y2}, color, 1);
    }
    imwrite("../img/src->tpl.png", dst);
}

void Match::transform_back(Mat &dst, int pad){
    if(!dst.empty()){
        src_img = dst;
    }
    if(pad == 0){
        warpAffine(src_img, dst, transform, dst.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 255, 0));
        return;
    }
    
    // make border, transform then crop img
    Mat pad_img;
    copyMakeBorder(src_img, pad_img, pad, pad, pad, pad, BORDER_CONSTANT, Scalar(0, 0, 255));
    warpAffine(pad_img, pad_img, transform, pad_img.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 255, 0));
    dst = pad_img;

    Mat v1 = (Mat_<double>(2, 3) << pad, pad, 1, src_img.cols + pad, src_img.rows + pad, 1); 
    Mat v2 = transform * v1.t();
    cout << "vertex " << v1 << " \n" << v2 << endl;
    cout << pad_img.size() << endl;
    
    int x1 = round(v2.at<double>(0, 0));
    int y1 = round(v2.at<double>(1, 0));
    int w = round(v2.at<double>(0, 1) - v2.at<double>(0, 0));
    int h = round(v2.at<double>(1, 1) - v2.at<double>(1, 0));
    
    assert(x1 >= 0 && y1 >= 0 && (x1 + w) < pad_img.cols && (y1 + h) < pad_img.rows);
    dst = pad_img(Rect{x1, y1, w, h});
    cout << dst.size() << endl;
    // show
    imwrite("../img/img_pad.jpg", pad_img);
}

void Match::draw_point(Mat &img, vector<Mask_Points> &points){
    RNG rng(12345);
    for(int i = 0; i < points.size(); ++i){
        Mask_Points &mp = points[i];
        Scalar color = Scalar(rng.uniform(100, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        for(int j = 0; j < mp.size(); ++j){
            Point2f &p = mp[j].pt;
            circle(img, {(int)p.x, (int)p.y}, 1, color, 2);
        }
    }
}