#ifndef _GMS_H_
#define _GMS_H_

#include "opencv2/opencv.hpp"
#include <vector>

// 8 possible rotation and each one is 3 X 3
const int mRotationPatterns[8][9] = {
    {
        1,2,3,
        4,5,6,
        7,8,9
    },
    {
        4,1,2,
        7,5,3,
        8,9,6
    },
    {
        7,4,1,
        8,5,2,
        9,6,3
    },
    {
        8,7,4,
        9,5,1,
        6,3,2
    },
    {
        9,8,7,
        6,5,4,
        3,2,1
    },
    {
        6,9,8,
        3,5,7,
        2,1,4
    },
    {
        3,6,9,
        2,5,8,
        1,4,7
    },
    {
        2,3,6,
        1,5,9,
        4,7,8
    }
};

// 5 level scales
const double mScaleRatios[5] = { 1.0, 1.0 / 2, 1.0 / std::sqrt(2.0), std::sqrt(2.0), 2.0 };

class GMSMatcher
{
public:
    // OpenCV Keypoints & Correspond Image Size & Nearest Neighbor Matches
    GMSMatcher(const std::vector<cv::KeyPoint>& vkp1, const cv::Size& size1, const std::vector<cv::KeyPoint>& vkp2, const cv::Size& size2,
               const std::vector<cv::DMatch>& vDMatches, const double thresholdFactor) : mThresholdFactor(thresholdFactor)
    {
        // Input initialize
        normalizePoints(vkp1, size1, mvP1);
        normalizePoints(vkp2, size2, mvP2);

        mNumberMatches = vDMatches.size();
        convertMatches(vDMatches, mvMatches);
        // Grid initialize
        mGridSizeLeft = cv::Size(20, 20); // this should be changed with image size?
        mGridNumberLeft = mGridSizeLeft.width * mGridSizeLeft.height;

        // Initialize the neighbor of left grid
        mGridNeighborLeft = cv::Mat::zeros(mGridNumberLeft, 9, CV_32SC1);
        initalizeNeighbors(mGridNeighborLeft, mGridSizeLeft);
    }

    ~GMSMatcher() {}

    // Get Inlier Mask
    // Return number of inliers
    int getInlierMask(std::vector<bool> &vbInliers, const bool withRotation = false, const bool withScale = false);

private:
    // Normalized Points
    std::vector<cv::Point2f> mvP1, mvP2;

    // Matches
    std::vector<std::pair<int, int> > mvMatches;

    // Number of Matches
    size_t mNumberMatches;

    // Grid Size
    cv::Size mGridSizeLeft, mGridSizeRight;
    int mGridNumberLeft;
    int mGridNumberRight;

    // x      : left grid idx
    // y      : right grid idx
    // value  : how many matches from idx_left to idx_right
    cv::Mat mMotionStatistics;

    //
    std::vector<int> mNumberPointsInPerCellLeft;

    // Inldex  : grid_idx_left
    // Value   : grid_idx_right
    std::vector<int> mCellPairs;

    // Every Matches has a cell-pair
    // first  : grid_idx_left
    // second : grid_idx_right
    std::vector<std::pair<int, int> > mvMatchPairs;

    // Inlier Mask for output
    std::vector<bool> mvbInlierMask;

    //
    cv::Mat mGridNeighborLeft;
    cv::Mat mGridNeighborRight;

    double mThresholdFactor;


    // Assign Matches to Cell Pairs
    void assignMatchPairs(const int GridType);

    void convertMatches(const std::vector<cv::DMatch> &vDMatches, std::vector<std::pair<int, int> > &vMatches);

    int getGridIndexLeft(const cv::Point2f &pt, const int type);

    int getGridIndexRight(const cv::Point2f &pt);

    std::vector<int> getNB9(const int idx, const cv::Size& GridSize);

    void initalizeNeighbors(cv::Mat &neighbor, const cv::Size& GridSize);

    void normalizePoints(const std::vector<cv::KeyPoint> &kp, const cv::Size &size, std::vector<cv::Point2f> &npts);

    // Run
    int run(const int rotationType);

    void setScale(const int scale);

    // Verify Cell Pairs
    void verifyCellPairs(const int rotationType);
};

#endif