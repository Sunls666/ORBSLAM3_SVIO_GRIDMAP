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
 * 例： ./Examples/Stereo-Inertial/stereo_inertial_kitti 
 ./Vocabulary/ORBvoc.txt 
 ./Examples/Stereo-Inertial/KITTI05-10.yaml
 mydataset/kitti-odometry/data_odometry_color/dataset/sequences/07 
 mydataset/kitti-raw/Residential/2011_09_30/2011_09_30_drive_0027_extract/oxts/data 
 mydataset/kitti/07/times_imu100hz.txt
 * 
 * 注：(1)kitti-odometry的图像序列07对应kitti-raw/Residential/2011_09_30/2011_09_30_drive_0027_sync，注意是sync不是extract，
 *    sync是已对齐和矫正的数据，extract未对齐未校正的原始数据，但是odometry的图像通常会少了raw的最后几帧。
 *   (2)IMU数据使用的是extract，IMU时间序列由extract中的IMU时间减去sync中图像的第一帧时间得来。
 *
 * 
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>
#include <ctime>
#include <sstream>

#include<opencv2/core/core.hpp>

#include<System.h>
#include "ImuTypes.h"
#include "Optimizer.h"

using namespace std;

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps);
void LoadIMU(const string &strImuPath, const string &strImuTimesPath, vector<double> &vTimestamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro);

int main(int argc, char **argv)
{
    if(argc != 6)
    {
        cerr << endl << "Usage: ./stereo_inertial_kitti path_to_vocabulary path_to_settings path_to_sequence path_to_imudata path_to_imutime" << endl;
        return 1;
    }

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
    }

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
    SLAM.SaveTrajectoryKITTI("/home/sun/traj_kitti.txt");

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
