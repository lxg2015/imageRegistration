#ifndef _Match_H_
#define _Match_H_

#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <vector>
#include <string>

class Match{
public:
    Match();
    typedef std::vector<cv::KeyPoint> Mask_Points;
    
    void load_tpl(std::string name);
    void load_src(std::string name);
    void compute_simi();
    void transform_back(cv::Mat &dst);
    void get_feat_desc(cv::Mat &img, cv::Mat &mask, cv::Mat &label, std::vector<Mask_Points> &feats, std::vector<cv::Mat> &desc);
    
    void draw_point(cv::Mat &img, std::vector<Mask_Points> &points);

private:
    int num_area;
    int num_points;
    int method; // CV_RANSAC or CV_LMEDS(least median robust method Ransac > 50%) RHO algorithm ?
    double outlier_thresh; // ransac reproject error threshold 3
    size_t max_iter; // ransac max iters maxium=2000
    double conf;  // estimated transformation 0.95->0.99 
    double distance_thresh; // match distance thresh
    bool use_gms; // use gms to exclude wrong match

    // detector
    cv::Ptr<cv::Feature2D> feature;
    std::vector<Mask_Points> tpl_feats;
    std::vector<Mask_Points> src_feats;

    cv::Mat tpl_mask;
    cv::Mat tpl_label;
    cv::Mat tpl_img;
    
    cv::Mat src_mask;
    cv::Mat src_img;
    
    // descriptor
    typedef std::vector<cv::Mat> Mask_Descs;
    std::vector<Mask_Descs> descs;
    std::vector<cv::Mat> tpl_descs;
    std::vector<cv::Mat> src_descs;

    // match
    cv::Ptr<cv::DescriptorMatcher> matcher;
    std::vector<cv::DMatch> match_points;
    cv::Mat transform;
};


#endif