#include "gms.h"

using namespace std;
using namespace cv;

// Normalize Key Points to Range(0 - 1)
void GMSMatcher::normalizePoints(const vector<KeyPoint> &kp, const Size &size, vector<Point2f> &npts)
{
    const size_t numP = kp.size();
    const int width   = size.width;
    const int height  = size.height;
    npts.resize(numP);

    for (size_t i = 0; i < numP; i++)
    {
        npts[i].x = kp[i].pt.x / width;
        npts[i].y = kp[i].pt.y / height;
    }
}

// Convert OpenCV DMatch to Match (pair<int, int>)
void GMSMatcher::convertMatches(const vector<DMatch> &vDMatches, vector<pair<int, int> > &vMatches)
{
    vMatches.resize(mNumberMatches);
    for (size_t i = 0; i < mNumberMatches; i++){
        vMatches[i] = pair<int, int>(vDMatches[i].queryIdx, vDMatches[i].trainIdx);
    }
}

// Get Neighbor 9 coord around region idx
vector<int> GMSMatcher::getNB9(const int idx, const Size& gridSize)
{
    vector<int> NB9(9, -1);

    int idx_x = idx % gridSize.width;
    int idx_y = idx / gridSize.width;

    for (int yi = -1; yi <= 1; yi++)
    {
        for (int xi = -1; xi <= 1; xi++)
        {
            int idx_xx = idx_x + xi;
            int idx_yy = idx_y + yi;

            if (idx_xx < 0 || idx_xx >= gridSize.width || idx_yy < 0 || idx_yy >= gridSize.height)
                continue;

            NB9[xi + 4 + yi * 3] = idx_xx + idx_yy * gridSize.width;
        }
    }
    return NB9;
}

// get neighbor coord range for each region
void GMSMatcher::initalizeNeighbors(Mat &neighbor, const Size& gridSize)
{
    for (int i = 0; i < neighbor.rows; i++)
    {
        vector<int> NB9 = getNB9(i, gridSize);
        int *data = neighbor.ptr<int>(i);
        memcpy(data, &NB9[0], sizeof(int) * 9);
    }
}

// match
int GMSMatcher::getInlierMask(vector<bool> &vbInliers, const bool withRotation, const bool withScale)
{
    int max_inlier = 0;
    if (!withScale && !withRotation)
    {
        setScale(0);
        
        max_inlier = run(1);
        vbInliers = mvbInlierMask;
        return max_inlier;
    }

    if (withRotation && withScale)
    {
        for (int scale = 0; scale < 5; scale++)
        {
            setScale(scale);
            for (int rotationType = 1; rotationType <= 8; rotationType++)
            {
                int num_inlier = run(rotationType);

                if (num_inlier > max_inlier)
                {
                    vbInliers = mvbInlierMask;
                    max_inlier = num_inlier;
                }
            }
        }
        return max_inlier;
    }

    if (withRotation && !withScale)
    {
        setScale(0);
        for (int rotationType = 1; rotationType <= 8; rotationType++)
        {
            int num_inlier = run(rotationType);

            if (num_inlier > max_inlier)
            {
                vbInliers = mvbInlierMask;
                max_inlier = num_inlier;
            }
        }
        return max_inlier;
    }

    if (!withRotation && withScale)
    {
        for (int scale = 0; scale < 5; scale++)
        {
            setScale(scale);
            int num_inlier = run(1);

            if (num_inlier > max_inlier)
            {
                vbInliers = mvbInlierMask;
                max_inlier = num_inlier;
            }

        }
        return max_inlier;
    }
    return max_inlier;
}

// get grid right scale neighbor coord range
void GMSMatcher::setScale(const int scale)
{
    // Set Scale
    mGridSizeRight.width = cvRound(mGridSizeLeft.width  * mScaleRatios[scale]);
    mGridSizeRight.height = cvRound(mGridSizeLeft.height * mScaleRatios[scale]);
    mGridNumberRight = mGridSizeRight.width * mGridSizeRight.height;

    // Initialize the neighbor of right grid
    mGridNeighborRight = Mat::zeros(mGridNumberRight, 9, CV_32SC1);
    initalizeNeighbors(mGridNeighborRight, mGridSizeRight);
}

// assign match
int GMSMatcher::run(const int rotationType)
{
    mvbInlierMask.assign(mNumberMatches, false);

    // Initialize Motion Statisctics
    mMotionStatistics = Mat::zeros(mGridNumberLeft, mGridNumberRight, CV_32SC1);
    mvMatchPairs.assign(mNumberMatches, pair<int, int>(0, 0));

    // 3.1.a
    for (int gridType = 1; gridType <= 4; gridType++)
    {
        // initialize
        mMotionStatistics.setTo(0);
        mCellPairs.assign(mGridNumberLeft, -1);
        mNumberPointsInPerCellLeft.assign(mGridNumberLeft, 0);

        assignMatchPairs(gridType);  // to avoid point lie in grid edge 
        verifyCellPairs(rotationType);

        // Mark inliers
        for (size_t i = 0; i < mNumberMatches; i++)
        {
            if (mCellPairs[mvMatchPairs[i].first] == mvMatchPairs[i].second)
                mvbInlierMask[i] = true;
        }
    }

    return (int) count(mvbInlierMask.begin(), mvbInlierMask.end(), true); //number of inliers
}

// assign match left point and point pair to grid cell
void GMSMatcher::assignMatchPairs(const int gridType)
{
    for (size_t i = 0; i < mNumberMatches; i++)
    {
        Point2f &lp = mvP1[mvMatches[i].first];
        Point2f &rp = mvP2[mvMatches[i].second];
        int lgidx = mvMatchPairs[i].first = getGridIndexLeft(lp, gridType);
        int rgidx = -1;

        if (gridType == 1)
        {
            rgidx = mvMatchPairs[i].second = getGridIndexRight(rp);
        }
        else
        {
            rgidx = mvMatchPairs[i].second;
        }

        if (lgidx < 0 || rgidx < 0) continue;

        mMotionStatistics.at<int>(lgidx, rgidx)++;
        mNumberPointsInPerCellLeft[lgidx]++;
    }
}

// 
void GMSMatcher::verifyCellPairs(const int rotationType)
{
    const int *CurrentRP = mRotationPatterns[rotationType - 1];

    for (int i = 0; i < mGridNumberLeft; i++)
    {
        if (sum(mMotionStatistics.row(i))[0] == 0)
        {
            mCellPairs[i] = -1;
            continue;
        }

        int max_number = 0;
        for (int j = 0; j < mGridNumberRight; j++)
        {
            int *value = mMotionStatistics.ptr<int>(i);
            if (value[j] > max_number)
            {
                mCellPairs[i] = j;
                max_number = value[j];
            }
        }

        int idx_grid_rt = mCellPairs[i];

        const int *NB9_lt = mGridNeighborLeft.ptr<int>(i);
        const int *NB9_rt = mGridNeighborRight.ptr<int>(idx_grid_rt);

        int score = 0;
        double thresh = 0;
        int numpair = 0;

        for (size_t j = 0; j < 9; j++)
        {
            int ll = NB9_lt[j];
            int rr = NB9_rt[CurrentRP[j] - 1];
            if (ll == -1 || rr == -1)
                continue;

            score += mMotionStatistics.at<int>(ll, rr);
            thresh += mNumberPointsInPerCellLeft[ll];
            numpair++;
        }

        thresh = mThresholdFactor * std::sqrt(thresh / numpair);

        if (score < thresh)
            mCellPairs[i] = -2;
    }
}

int GMSMatcher::getGridIndexLeft(const Point2f &pt, const int type)
{
    int x = 0, y = 0;

    if (type == 1) {
        x = cvFloor(pt.x * mGridSizeLeft.width);
        y = cvFloor(pt.y * mGridSizeLeft.height);
    }

    if (type == 2) {
        x = cvFloor(pt.x * mGridSizeLeft.width + 0.5);
        y = cvFloor(pt.y * mGridSizeLeft.height);
    }

    if (type == 3) {
        x = cvFloor(pt.x * mGridSizeLeft.width);
        y = cvFloor(pt.y * mGridSizeLeft.height + 0.5);
    }

    if (type == 4) {
        x = cvFloor(pt.x * mGridSizeLeft.width + 0.5);
        y = cvFloor(pt.y * mGridSizeLeft.height + 0.5);
    }


    if (x >= mGridSizeLeft.width || y >= mGridSizeLeft.height)
        return -1;

    return x + y * mGridSizeLeft.width;
}

int GMSMatcher::getGridIndexRight(const Point2f &pt)
{
    int x = cvFloor(pt.x * mGridSizeRight.width);
    int y = cvFloor(pt.y * mGridSizeRight.height);

    return x + y * mGridSizeRight.width;
}
