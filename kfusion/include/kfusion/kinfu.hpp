#pragma once

#include "types.hpp"
#include "cuda/tsdf_volume.hpp"
#include "cuda/projective_icp.hpp"
#include <vector>
#include <string>

namespace kfusion
{
    namespace cuda
    {
        KF_EXPORTS int getCudaEnabledDeviceCount();
        KF_EXPORTS void setDevice(int device);
        KF_EXPORTS std::string getDeviceName(int device);
        KF_EXPORTS bool checkIfPreFermiGPU(int device);
        KF_EXPORTS void printCudaDeviceInfo(int device);
        KF_EXPORTS void printShortCudaDeviceInfo(int device);
    }

    struct KF_EXPORTS KinFuParams
    {
        static KinFuParams default_params();

        int cols;  //pixels
        int rows;  //pixels

        Intr intr;  //Camera parameters

        Vec3i volume_dims; //number of voxels
        Vec3f volume_size; //meters
        Affine3f volume_pose; //meters, inital pose

        float bilateral_sigma_depth;   //meters
        float bilateral_sigma_spatial;   //pixels
        int   bilateral_kernel_size;   //pixels

        float icp_truncate_depth_dist; //meters
        float icp_dist_thres;          //meters
        float icp_angle_thres;         //radians
        std::vector<int> icp_iter_num; //iterations for level index 0,1,..,3

        float tsdf_min_camera_movement; //meters, integrate only if exceedes
        float tsdf_trunc_dist;             //meters;
        int tsdf_max_weight;               //frames

        float raycast_step_factor;   // in voxel sizes
        float gradient_delta_factor; // in voxel sizes

        Vec3f light_pose; //meters

		float icp_pose_angle_thres; // difference between sequential poses rotation
		float icp_pose_dist_thres;  // difference between sequential poses translation

    };

    class KF_EXPORTS KinFu
    {
    public:        
        typedef cv::Ptr<KinFu> Ptr;

        KinFu(const KinFuParams& params);

        const KinFuParams& params() const;
        KinFuParams& params();

        const cuda::TsdfVolume& tsdf() const;
        cuda::TsdfVolume& tsdf();

        const cuda::ProjectiveICP& icp() const;
        cuda::ProjectiveICP& icp();

        void reset();

		// returns false only when failed
        bool operator()(const cuda::Depth& dpeth, float& angle_diff, float& dist_diff, const cuda::Image& image = cuda::Image());

        void renderImage(cuda::Image& image, int flags = 0);
        void renderImage(cuda::Image& image, const Affine3f& pose, int flags = 0);

        Affine3f getCameraPose (int time = -1) const;

		bool extractMesh(float*& buffer, int& num_points) const;
		bool extractMesh(float*& vbuffer, int& num_points, int*& ibuffer, int& num_indices) const;

    private:
        void allocate_buffers();

        int frame_counter_;
        KinFuParams params_;

        std::vector<Affine3f> poses_;

        cuda::Dists dists_;
        cuda::Frame curr_, prev_;

        cuda::Cloud points_;
        cuda::Normals normals_;
        cuda::Depth depths_;

        cv::Ptr<cuda::TsdfVolume> volume_;
        cv::Ptr<cuda::ProjectiveICP> icp_;
    };
}
