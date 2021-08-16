#ifndef SLAM_MAPDB_HPP
#define SLAM_MAPDB_HPP

#include "id.hpp"
#include "keyframe.hpp"
#include "loop_closer.hpp"
#include "map_point.hpp"
#include "../api/slam_map_point_record.hpp"

#include <map>
#include <set>

namespace slam {

class MapDB {
public:

    MapDB() :
        prevPose(Eigen::Matrix4d::Identity()),
        prevInputPose(Eigen::Matrix4d::Identity()),
        prevUncertainty(Eigen::Matrix<double, 3, 6>::Zero())
    {};
    MapDB(const MapDB &mapDB); // Copy constructor
    MapDB(const MapDB &mapDB, const std::set<KfId> &activeKeyframes); // Copy constructor that only copies given frames

    std::map<KfId, std::shared_ptr<Keyframe>> keyframes;
    std::map<MpId, MapPoint> mapPoints;
    std::map<TrackId, MpId> trackIdToMapPoint;
    std::vector<LoopClosureEdge> loopClosureEdges;

    double firstKfTimestamp = -1.0;

    std::shared_ptr<Keyframe> insertNewKeyframeCandidate(
        std::unique_ptr<Keyframe> keyframe,
        bool keyframeDecision,
        const std::vector<slam::Pose> &poseTrail,
        const odometry::ParametersSlam &parameters);

    std::map<MpId, MapPoint>::iterator removeMapPoint(const MapPoint &mapPoint);

    MpId nextMpId();
    KfId lastKeyframeCandidateId() const { return lastKfCandidateId; }
    Keyframe *latestKeyframe() {
        if (lastKfId.v >= 0) {
            if (!keyframes.count(lastKfId)) return nullptr;
            return keyframes.at(lastKfId).get();
        }
        return nullptr;
    }
    std::pair<KfId, MpId> maxIds() const;

    void mergeMapPoints(MpId mpId1, MpId mpId2);

    Eigen::Matrix4d poseDifference(KfId kfId1, KfId kfId2) const;

    void updatePrevPose(const Keyframe &currentKeyframe, const Eigen::Matrix4d &inputPose);

    // Visualization stuff stored here for convenience.
    std::map<MapKf, LoopStage> loopStages;
    std::vector<KfId> adjacentKfIds;
    std::map<MpId, MapPointRecord> mapPointRecords;

    // Stuff for JSON SLAM map output.
    int outputInd = 0;
    std::map<KfId, std::string> cameraImagesLeft;
    std::map<KfId, std::string> cameraImagesRight;

private:
    Eigen::Matrix4d prevPose, prevInputPose, prevSmoothPose;
    Eigen::Matrix<double, 3, 6> prevUncertainty;
    int nextMp = 0;
    // id of the frame corresponding to prevPose & prevInput pose. May no longer exist
    KfId prevPoseKfId = KfId(-1);
    // id of the last inserted, thing, which may no longer exist
    KfId lastKfCandidateId = KfId(-1);
    // lastest keyframe ID. Should exist
    KfId lastKfId = KfId(-1);
};

using Atlas = std::vector<MapDB>;

const MapDB& getMapWithId(MapId mapId, const MapDB &mapDB, const Atlas &atlas);

} // namespace slam

#endif // SLAM_MAPDB_HPP
