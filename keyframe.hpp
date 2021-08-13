#ifndef SLAM_KEYFRAME_HPP
#define SLAM_KEYFRAME_HPP

#include <Eigen/StdVector>
#include <map>

#include <Eigen/Dense>
#include <DBoW2/BowVector.h>
#include <DBoW2/FeatureVector.h>
#include <opencv2/core/types.hpp>

#include "feature_search.hpp"
#include "id.hpp"
#include "bow_index.hpp"
#include "../api/slam.hpp"
#include "key_point.hpp"
#include "../tracker/camera.hpp"

namespace slam {

inline Eigen::Vector3d worldToCameraMatrixCameraCenter(const Eigen::Matrix4d &poseCW) {
    return -poseCW.topLeftCorner<3, 3>().transpose() * poseCW.block<3, 1>(0, 3);
}

class MapPoint;
class MapDB;
struct OrbExtractor;

struct MapperInput {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    std::shared_ptr<tracker::Image> frame;
    std::shared_ptr<tracker::Image> frameRight;
    std::vector<Feature> trackerFeatures;
    std::vector<slam::Pose> poseTrail;
    double t;

    // for debuggging & nicer visualization. Can be empty
    cv::Mat colorFrame;

    MapperInput();
    ~MapperInput();
};

/**
 * Keyframe information that can be shared between copies, accessed from
 * multiple threads
 */
struct KeyframeShared {
    std::shared_ptr<const tracker::Camera> camera;
    std::shared_ptr<const tracker::Camera> cameraRight;
    KeyPointVector keyPoints;
    std::unique_ptr<FeatureSearch> featureSearch; // For speeding up feature search.

    // debugging / visu
    cv::Mat imgDbg;
    std::vector<cv::Vec3b> colors;
    using StereoPointCloud =   std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>;
    std::shared_ptr<StereoPointCloud> stereoPointCloud;
    std::shared_ptr<std::vector<cv::Vec3b>> stereoPointCloudColor;

    // The bag-of-words representation of the image.
    // Size() of this map is about the same as number of keypoints, because typically
    // the vocabulary has much more words than there are keypoints (10^6 >> 10^4), so
    // most keypoints will be assigned to a unique word.
    // * std::map<WordId, WordValue> = std::map<unsigned int, double>
    DBoW2::BowVector bowVec;

    // List of keypoint ids for each node on a specific level of the vocabulary tree.
    // The higher the level, the larger and fewer the nodes (buckets) become.
    // Typically the level is set so that there are about 100 nodes.
    // Called "direct index" in the DBoW paper.
    // * std::map<NodeId, std::vector<unsigned int>>, where NodeId = unsigned int.
    DBoW2::FeatureVector bowFeatureVec;

    // make a copy, assuming most of the fields have not been populated yet
    // (assert fails if this used too late)
    std::unique_ptr<KeyframeShared> clone() const;
};

class Keyframe {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Keyframe(const MapperInput &mapperInput);

    Keyframe(const Keyframe &kf); // Copy constructor

    Keyframe();

    bool hasFeatureDescriptors() const { return hasFullFeatures; }

    void addFullFeatures(const MapperInput &mapperInput, OrbExtractor &orb);
    void addTrackerFeatures(const MapperInput &mapperInput);

    void addObservation(MpId mapPointId, KpId keyPointId);

    void eraseObservation(MpId mapPointId);

    /**
     *  Compute median depth of mapPoints observed in keyframe
     *
     *  @return median depth, if no triangulated MapPoint%s are observed return #defaultDepth
     */
    float computeMedianDepth(const MapDB &mapDB, float defaultDepth = 2.0);

    bool isInFrustum(const MapPoint &mp, float viewAngleLimitCos = 0.5) const;

    bool reproject(const Eigen::Vector3d &point, Eigen::Vector2f &reprojected) const;

    Eigen::Vector3d cameraCenter() const;
    Eigen::Vector3d smoothPoseCameraCenter() const;
    inline Eigen::Matrix3d cameraToWorldRotation() const {
        return poseCW.topLeftCorner<3, 3>().transpose();
    }

    void getFeaturesAround(const Eigen::Vector2f &point, float r, std::vector<size_t> &output);

    std::vector<KfId> getNeighbors(
        const MapDB &mapDB,
        int minCovisibilities = 1,
        bool triangulatedOnly = false
    ) const;

    // for visualizations
    const cv::Vec3b &getKeyPointColor(KpId kp) const;

    // Immutable properties that are shared among all copies of the keyframe
    std::shared_ptr<KeyframeShared> shared;

    KfId id;
    KfId previousKfId;
    KfId nextKfId;

    std::map<KpId, TrackId> keyPointToTrackId;

    // These are indexed by keypoint id value.
    std::vector<MpId> mapPoints;
    std::vector<float> keyPointDepth;

    // "CW" means world-to-camera, coming from notation like `p_C = T_CW * p_W`.
    // The main pose of the keyframe that is used for most computations and initializing
    // KF positions in non-linear optimization.
    Eigen::Matrix4d poseCW;

    // These poses are meant to form a smooth path so that differences of these
    // from consecutive keyframes can be used as odometry priors
    // Especially if the odometry result is very good, the unfiltered odometry
    // output poses can be used for this purpose.
    Eigen::Matrix4d smoothPoseCW;

    // Accumulated uncertainty (for position and orientation separately).
    // These are formed by summing covariance matrices that represent the uncertainty
    // between consecutive key frames, which corresponds to the assumption / simplification
    // that those noises are independent (which is not very accurate). However, this means
    // that these uncertainty matrices form a total order: if t1 >= t2, then
    // uncertainty2 - uncertainty1 is positive semidefinite. This simplifies the handling
    // of dropped keyframes as you can always just subtract these and still get a valid
    // covariance matrix
    Eigen::Matrix<double, 3, 6> uncertainty;

    double t;

    // True if ORB features have been computed, false if only
    // odometry-based info is present
    bool hasFullFeatures;

    cv::Mat cpuImageLeft;
    cv::Mat cpuImageRight;

    // For debugging
    std::string toString();
};

// OpenVSLAM helper functions
bool reprojectToImage(
    const tracker::Camera &camera,
    const Eigen::Matrix3d& rot_cw,
    const Eigen::Vector3d& trans_cw,
    const Eigen::Vector3d& pos_w,
    Eigen::Vector2d& reproj,
    float& x_right);

bool reprojectToBearing(
    const tracker::Camera &camera,
    const Eigen::Matrix3d& rot_cw,
    const Eigen::Vector3d& trans_cw,
    const Eigen::Vector3d& pos_w,
    Eigen::Vector3d& reproj);

/**
 * An ASCII visualization of keyframe statuses.
 *
 * @param status Function that tells what character should be printed for given keyframe id.
 * @param mapDB
 * @param len Width of the visualization in terminal characters.
 */
void asciiKeyframes(const std::function<char(KfId)> status, const MapDB &mapDB, int len);

} // namespace slam
#endif //SLAM_KEYFRAME_HPP
