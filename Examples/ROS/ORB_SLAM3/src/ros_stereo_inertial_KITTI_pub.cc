/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

/**
 * 说明
 * 该文件是参照stereo_kitti.cc和stereo_inertial_euroc.cc改的。
 * 参数：./stereo_inertial_kitti 词汇路径 配置文件路径 图像路径（含序列和时间） IMU路径(extract)  IMU时间路径。
 * 例： ./Examples/Stereo-Inertial/stereo_inertial_kitti ./Vocabulary/ORBvoc.txt ./Examples/Stereo-Inertial/KITTI05-10.yaml
 *      mydataset/kitti-odometry/data_odometry_color/dataset/sequences/07 
 *      mydataset/kitti-raw/Residential/2011_09_30/2011_09_30_drive_0027_extract/oxts/data mydataset/kitti/07/times_imu100hz.txt
 * 
 * 注：(1)kitti-odometry的图像序列07对应kitti-raw/Residential/2011_09_30/2011_09_30_drive_0027_sync，注意是sync不是extract，
 *    sync是已对齐和矫正的数据，extract未对齐未校正的原始数据，但是odometry的图像通常会少了raw的最后几帧。
 *   (2)IMU数据使用的是extract，IMU时间序列由extract中的IMU时间减去sync中图像的第一帧时间得来。
 *
 * 
*/

#include <unistd.h>
#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>
#include <time.h>
#include <ctime>
#include <sstream>

#include<ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include "sensor_msgs/PointCloud2.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/PoseArray.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <opencv2/core/core.hpp>

#include"../../../include/System.h"
#include "ImuTypes.h"
#include "Optimizer.h"


#include "MapPoint.h"
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <Converter.h>

using namespace std;

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps);
void LoadIMU(const string &strImuPath, const string &strImuTimesPath, vector<double> &vTimestamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro);

void publish(ORB_SLAM3::System &SLAM, ros::Publisher &pub_pts_and_pose,
	ros::Publisher &pub_all_kf_and_pts, int index);

class ImageGrabber{
public:
	ImageGrabber(ORB_SLAM3::System &_SLAM, ros::Publisher &_pub_pts_and_pose,
		ros::Publisher &_pub_all_kf_and_pts) :
		SLAM(_SLAM), pub_pts_and_pose(_pub_pts_and_pose),
		pub_all_kf_and_pts(_pub_all_kf_and_pts), index(0){}

	ORB_SLAM3::System &SLAM;
	ros::Publisher &pub_pts_and_pose;
	ros::Publisher &pub_all_kf_and_pts;
    int index = 0;
};

int all_pts_pub_gap = 1;
bool pub_all_pts = false;
int pub_count = 0;
double scalefactor = 0.1;

int main(int argc, char **argv)
{
    if(argc != 6)
    {
        cerr << endl << "Usage: ./stereo_inertial_kitti path_to_vocabulary path_to_settings path_to_sequence path_to_imudata path_to_imutime" << endl;
        return 1;
    }

    ros::init(argc, argv, "Stereo_Inertial_pub");
	ros::start();

   // JC Modified: IMU variable
   vector<cv::Point3f> vAcc, vGyro;
   vector<double> vTimestampsImu;
   int nImu;
   int first_imu=0;
   string pathImu = argv[4];
   string pathImuTimes = argv[5];

    // Retrieve paths to images
   cout << "Loading images for sequence " <<"...";
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimestampsCam;
    LoadImages(string(argv[3]), vstrImageLeft, vstrImageRight, vTimestampsCam);
   cout << "LOADED!" << endl;

   // JC Modified: IMU Load
   cout << "Loading IMU for sequence " << "...";
   LoadIMU(pathImu,pathImuTimes ,vTimestampsImu, vAcc, vGyro);
   cout << "LOADED!" << endl;
   nImu=vTimestampsImu.size();

    const int nImages = vstrImageLeft.size();
   if((nImages<=0)||(nImu<=0))
   {
      cerr << "ERROR: Failed to load images or IMU for sequence" << endl;
      return 1;
   }

    // JC Modified:
    // Find first imu to be considered, supposing imu measurements start first
    while(vTimestampsImu[first_imu]<=vTimestampsCam[0])
        first_imu++;
    first_imu--; // first imu measurement to be considered


    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::IMU_STEREO,true);

    ros::NodeHandle nodeHandler;
    ros::Publisher pub_pts_and_pose = nodeHandler.advertise<geometry_msgs::PoseArray>("pts_and_pose", 1000);
	ros::Publisher pub_all_kf_and_pts = nodeHandler.advertise<geometry_msgs::PoseArray>("all_kf_and_pts", 1000);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;   

    // Main loop
    cv::Mat imLeft, imRight;
    // JC Modified 
    vector<ORB_SLAM3::IMU::Point> vImuMeas;
    double t_rect = 0;
    int num_rect = 0;
    int proccIm = 0;
    cout << "Imudatas in the sequence: " << nImu << endl << endl;   

    ros::Rate loop_rate(5);

    for(int ni=0; ni<nImages; ni++)
    {
        // Read left and right images from file
        imLeft = cv::imread(vstrImageLeft[ni],cv::IMREAD_UNCHANGED);
        imRight = cv::imread(vstrImageRight[ni],cv::IMREAD_UNCHANGED);
        double tframe = vTimestampsCam[ni];

        if(imLeft.empty())
        {
            cerr << endl << "Failed to load left image at: "
                 << string(vstrImageLeft[ni]) << endl;
            return 1;
        }

        // JC Modified 
        if(imRight.empty())
        {
            cerr << endl << "Failed to load right image at: "
                    << string(vstrImageRight[ni]) << endl;
            return 1;
        }

        // JC Modified: Load imu measurements from previous frame
        //        将相邻帧之间的IMU数据存入vImuMeas，用于Track
        vImuMeas.clear();
        if(ni>0)
            while(vTimestampsImu[first_imu]<=vTimestampsCam[ni]) // while(vTimestampsImu[first_imu]<=vTimestampsCam[ni])
            {
                vImuMeas.push_back(ORB_SLAM3::IMU::Point(vAcc[first_imu].x,vAcc[first_imu].y,vAcc[first_imu].z,
                                                            vGyro[first_imu].x,vGyro[first_imu].y,vGyro[first_imu].z,
                                                            vTimestampsImu[first_imu]));
                first_imu++;
            }


#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // JC Modified：Pass the images to the SLAM system
        // SLAM.TrackStereo(imLeft,imRight,tframe);
        SLAM.TrackStereo(imLeft,imRight,tframe,vImuMeas);
        publish(SLAM, pub_pts_and_pose, pub_all_kf_and_pts, ni);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestampsCam[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestampsCam[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);

        ros::spinOnce();
		loop_rate.sleep();
		if (!ros::ok()){ break; }
    }

    mkdir("results", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	SLAM.getMap()->GetCurrentMap()->Save("results//map_pts_out.obj");
	SLAM.getMap()->GetCurrentMap()->SaveWithTimestamps("results//map_pts_and_keyframes.txt");

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveTrajectoryKITTI("/home/sun/svio_kitti.txt");

    

    return 0;
}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_2/";
    string strPrefixRight = strPathToSequence + "/image_3/";

    const int nTimes = vTimestamps.size();
    vstrImageLeft.resize(nTimes);
    vstrImageRight.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
        vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
    }
}

void LoadIMU(const string &strImuPath, const string &strImuTimesPath, vector<double> &vTimestamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro)
{
    ifstream fTimes;
    fTimes.open(strImuTimesPath.c_str());
    vTimestamps.reserve(100000);

    // JC Modified 读取时间
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    const int nTimes = vTimestamps.size();


    // 遍历每个IMU数据文件，将数据存入vAcc、vGyro
    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(10) << i;
        string pathtemp = strImuPath + "/"+ss.str() + ".txt";
        ifstream fin ( pathtemp );
        if ( !fin )
        {
            cout<<"imudata file "<<pathtemp<<" does not exist."<<endl;
            return ;
        }
        //取12-14、18-20列数据作加速度和路径
        string s;
        getline(fin,s);
        if(!s.empty())
        {
            string item;
            size_t pos = 0;
            double data[23];
            int count = 0;
            while ((pos = s.find(' ')) != string::npos) {
                item = s.substr(0, pos);
                data[count++] = stod(item);
                s.erase(0, pos + 1);
            }
            item = s.substr(0, pos);
            data[6] = stod(item);

            vAcc.push_back(cv::Point3f(data[11],data[12],data[13]));
            vGyro.push_back(cv::Point3f(data[17],data[18],data[19]));
        }

    }
}

void publish(ORB_SLAM3::System &SLAM, ros::Publisher &pub_pts_and_pose,
	ros::Publisher &pub_all_kf_and_pts, int index) 
    {
	if (all_pts_pub_gap>0 && pub_count >= all_pts_pub_gap) {
		pub_all_pts = true;
		pub_count = 0;
	}
	if (pub_all_pts || SLAM.getLoopClosing()->loop_detected || SLAM.getTracker()->loop_detected) 
	{
		pub_all_pts = SLAM.getTracker()->loop_detected = SLAM.getLoopClosing()->loop_detected = false;
        /* PoseArray  位姿序列：
        （1）std_msgs/Header header
        （2）geometry_msgs/Pose[] poses*/
		geometry_msgs::PoseArray kf_pt_array;
		vector<ORB_SLAM3::KeyFrame*> key_frames = SLAM.getMap()->GetCurrentMap()->GetAllKeyFrames();
		//! placeholder for number of keyframes
        /*geometry_msgs::Pose()  位姿：
           （1）geometry_msgs/Point position 位置 
           （2）geometry_msgs/Quaternion orientation 姿态，即方向
        */
		kf_pt_array.poses.push_back(geometry_msgs::Pose());
		sort(key_frames.begin(), key_frames.end(), ORB_SLAM3::KeyFrame::lId);
		unsigned int n_kf = 0;

        //遍历关键帧
		for (auto key_frame : key_frames) 
        {
			// pKF->SetPose(pKF->GetPose()*Two);

			if (key_frame->isBad())
				continue;

			cv::Mat R = key_frame->GetRotation().t();
			vector<float> q = ORB_SLAM3::Converter::toQuaternion(R);
			cv::Mat twc = key_frame->GetCameraCenter();
            cv::Mat Pcc = key_frame->GetStereoCenter();
			geometry_msgs::Pose kf_pose;

            //kf_pose：关键帧的位姿(双目+IMU需要对调position的y和z，不知道原因)
			kf_pose.position.x = Pcc.at<float>(0)*scalefactor;
			kf_pose.position.y = Pcc.at<float>(2)*scalefactor;
			kf_pose.position.z = -Pcc.at<float>(1)*scalefactor;
			kf_pose.orientation.x = q[0];
			kf_pose.orientation.y = q[1];
			kf_pose.orientation.z = q[2];
			kf_pose.orientation.w = q[3];
			kf_pt_array.poses.push_back(kf_pose);

			unsigned int n_pts_id = kf_pt_array.poses.size(); 
			//! placeholder for number of points
			kf_pt_array.poses.push_back(geometry_msgs::Pose());
			std::set<ORB_SLAM3::MapPoint*> map_points = key_frame->GetMapPoints();
			unsigned int n_pts = 0;   //初始化地图点个数为0
            //遍历当前关键帧下的地图点
			for (auto map_pt : map_points)
            {
				if (!map_pt || map_pt->isBad()) {
					//printf("Point %d is bad\n", pt_id);
					continue;
				}
				cv::Mat pt_pose = map_pt->GetWorldPos();   //地图点的世界坐标
				if (pt_pose.empty()) {
					//printf("World position for point %d is empty\n", pt_id);
					continue;
				}
				geometry_msgs::Pose curr_pt;
				//printf("wp size: %d, %d\n", wp.rows, wp.cols);
				//pcl_cloud->push_back(pcl::PointXYZ(wp.at<float>(0), wp.at<float>(1), wp.at<float>(2)));

                //地图点的位置
				curr_pt.position.x = pt_pose.at<float>(0)*scalefactor;
				curr_pt.position.y = pt_pose.at<float>(2)*scalefactor;
				curr_pt.position.z = -pt_pose.at<float>(1)*scalefactor;
				kf_pt_array.poses.push_back(curr_pt);
				++n_pts;
			}
            //当前关键帧的地图点遍历结束

            /*为什么pose等于当前关键帧的地图点的个数？
            n_pts_id=目前所有遍历过的关键帧+目前所有遍历过的地图点的总个数
            是为了记录n_pts???
            */
			geometry_msgs::Pose n_pts_msg;
			n_pts_msg.position.x = n_pts_msg.position.y = n_pts_msg.position.z = n_pts;
			kf_pt_array.poses[n_pts_id] = n_pts_msg;
			++n_kf;
		}
		geometry_msgs::Pose n_kf_msg;
		n_kf_msg.position.x = n_kf_msg.position.y = n_kf_msg.position.z = n_kf;
		kf_pt_array.poses[0] = n_kf_msg;  //记录n_kf？？？
        /*header包括三个内容
        uint32 seq
        time stamp
        string frame_id
        */
		kf_pt_array.header.frame_id = "1";
		kf_pt_array.header.seq = index + 1;
		printf("Publishing data for %u keyfranmes\n", n_kf);
		pub_all_kf_and_pts.publish(kf_pt_array);
	}
    
	else if (SLAM.getTracker()->mCurrentFrame.is_keyframe) 
    {
		++pub_count;
		SLAM.getTracker()->mCurrentFrame.is_keyframe = false;
		ORB_SLAM3::KeyFrame* pKF = SLAM.getTracker()->mCurrentFrame.mpReferenceKF;   //pKF是参考关键帧

		cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);    //eye：单位矩阵

		// If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
		//while (pKF->isBad())
		//{
		//	Trw = Trw*pKF->mTcp;
		//	pKF = pKF->GetParent();
		//}

		vector<ORB_SLAM3::KeyFrame*> vpKFs = SLAM.getMap()->GetCurrentMap()->GetAllKeyFrames();   //获取所有关键帧
		sort(vpKFs.begin(), vpKFs.end(), ORB_SLAM3::KeyFrame::lId);  //排序规则是ORB_SLAM3::KeyFrame::lId，mnId从小到大

		// Transform all keyframes so that the first keyframe is at the origin.
		// After a loop closure the first keyframe might not be at the origin.
		cv::Mat Two = vpKFs[0]->GetPoseInverse();  //GetPoseInverse()获取位姿的逆，这里获取第0帧的位姿的逆
        
		Trw = Trw*pKF->GetPose()*Two;    //参考关键帧相对于第0帧的位姿
		cv::Mat lit = SLAM.getTracker()->mlRelativeFramePoses.back();   //当前帧相对其参考关键帧的相对变换矩阵
		cv::Mat Tcw = lit*Trw;     //当前帧到第0帧的变换矩阵
        /*
        rowRange colRange 包括左边界，但不包括右边界
        t()是转置矩阵
        */
		cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
		cv::Mat twc = -Rwc*Tcw.rowRange(0, 3).col(3);
        cv::Mat Pcc = pKF->GetStereoCenter();

		vector<float> q = ORB_SLAM3::Converter::toQuaternion(Rwc);
		//geometry_msgs::Pose camera_pose;
		//std::vector<ORB_SLAM3::MapPoint*> map_points = SLAM.getMap()->GetCurrentMap()->GetAllMapPoints();
		std::vector<ORB_SLAM3::MapPoint*> map_points = SLAM.GetTrackedMapPoints();
		int n_map_pts = map_points.size();

		//printf("n_map_pts: %d\n", n_map_pts);

		//pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);

		geometry_msgs::PoseArray pt_array;
		//pt_array.poses.resize(n_map_pts + 1);

		geometry_msgs::Pose camera_pose;

		camera_pose.position.x = Pcc.at<float>(0)*scalefactor;
		camera_pose.position.y = Pcc.at<float>(2)*scalefactor;
		camera_pose.position.z = -Pcc.at<float>(1)*scalefactor;

		camera_pose.orientation.x = q[0];
		camera_pose.orientation.y = q[1];
		camera_pose.orientation.z = q[2];
		camera_pose.orientation.w = q[3];

		pt_array.poses.push_back(camera_pose);

		//printf("Done getting camera pose\n");

		for (int pt_id = 1; pt_id <= n_map_pts; ++pt_id){

			if (!map_points[pt_id - 1] || map_points[pt_id - 1]->isBad()) {
				//printf("Point %d is bad\n", pt_id);
				continue;
			}
			cv::Mat wp = map_points[pt_id - 1]->GetWorldPos();

			if (wp.empty()) {
				//printf("World position for point %d is empty\n", pt_id);
				continue;
			}
			geometry_msgs::Pose curr_pt;
			//printf("wp size: %d, %d\n", wp.rows, wp.cols);
			//pcl_cloud->push_back(pcl::PointXYZ(wp.at<float>(0), wp.at<float>(1), wp.at<float>(2)));
			curr_pt.position.x = wp.at<float>(0)*scalefactor;
			curr_pt.position.y = wp.at<float>(2)*scalefactor;
			curr_pt.position.z = -wp.at<float>(1)*scalefactor;
			pt_array.poses.push_back(curr_pt);
			//printf("Done getting map point %d\n", pt_id);
		}
		//sensor_msgs::PointCloud2 ros_cloud;
		//pcl::toROSMsg(*pcl_cloud, ros_cloud);
		//ros_cloud.header.index = "1";
		//ros_cloud.header.index = ni;

		//printf("valid map pts: %lu\n", pt_array.poses.size()-1);

		//printf("ros_cloud size: %d x %d\n", ros_cloud.height, ros_cloud.width);
		//pub_cloud.publish(ros_cloud);
		pt_array.header.frame_id = "1";
		pt_array.header.seq = index + 1;
		pub_pts_and_pose.publish(pt_array);
		//pub_kf.publish(camera_pose);
	}
}
