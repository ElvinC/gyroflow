#include <sstream>
#include <string>
#include <fstream>
#include <time.h>
#include <thread>
#include <opencv2\opencv.hpp>
#include "json.hpp"

using json = nlohmann::json;

cv::Mat stabRotMatFromQuat(float w, float i, float j, float k);

using namespace cv;

class ParallelUndistort : public cv::ParallelLoopBody {
public:
    ParallelUndistort(cv::Mat& map1, cv::Mat& map2, cv::Mat K, cv::Mat D, cv::Mat R, cv::Mat P, const cv::Size& size, int m1type)
        :m_map1(map1), m_map2(map2), m_K(K), m_D(D), m_R(R), m_P(P), m_size(size), m_m1type(m1type) {

        CV_Assert(m_m1type == CV_16SC2 || m_m1type == CV_32F || m_m1type <= 0);
        m_map1.create(m_size, m1type <= 0 ? CV_16SC2 : m1type);
        m_map2.create(m_size, m_map1.type() == CV_16SC2 ? CV_16UC1 : CV_32F);

        CV_Assert((m_K.depth() == CV_32F || m_K.depth() == CV_64F) && (m_D.depth() == CV_32F || m_D.depth() == CV_64F));
        CV_Assert((m_P.empty() || m_P.depth() == CV_32F || m_P.depth() == CV_64F) && (m_R.empty() || m_R.depth() == CV_32F || m_R.depth() == CV_64F));
        CV_Assert(m_K.size() == cv::Size(3, 3) && (m_D.empty() || m_D.total() == 4));
        CV_Assert(m_R.empty() || m_R.size() == cv::Size(3, 3) || m_R.total() * m_R.channels() == 3);
        CV_Assert(m_P.empty() || m_P.size() == cv::Size(3, 3) || m_P.size() == cv::Size(4, 3));


        if (m_K.depth() == CV_32F)
        {
            camMat = m_K;
            f = Vec2f(camMat(0, 0), camMat(1, 1));
            c = Vec2f(camMat(0, 2), camMat(1, 2));
        }
        else
        {
            camMat = m_K;
            f = Vec2d(camMat(0, 0), camMat(1, 1));
            c = Vec2d(camMat(0, 2), camMat(1, 2));
        }


        k = Vec4d::all(0);
        if (!m_D.empty())
            k = m_D.depth() == CV_32F ? (Vec4d)*m_D.ptr<Vec4f>() : *m_D.ptr<Vec4d>();

        cv::Matx33d RR = cv::Matx33d::eye();
        if (!m_R.empty() && m_R.total() * m_R.channels() == 3)
        {
            cv::Vec3d rvec;
            m_R.convertTo(rvec, CV_64F);
            RR = Affine3d(rvec).rotation();
        }
        else if (!m_R.empty() && m_R.size() == Size(3, 3))
            m_R.convertTo(RR, CV_64F);

        cv::Matx33d PP = cv::Matx33d::eye();
        if (!m_P.empty())
            m_P.colRange(0, 3).convertTo(PP, CV_64F);

        iR = (PP * RR).inv(cv::DECOMP_SVD);

    }

    virtual void operator () (const cv::Range& range) const CV_OVERRIDE {

        for (int i = range.start; i < range.end; ++i)
        {
            float* m1f = m_map1.ptr<float>(i);
            float* m2f = m_map2.ptr<float>(i);
            short* m1 = (short*)m1f;
            ushort* m2 = (ushort*)m2f;

            double _x = i * iR(0, 1) + iR(0, 2),
                _y = i * iR(1, 1) + iR(1, 2),
                _w = i * iR(2, 1) + iR(2, 2);

            for (int j = 0; j < m_size.width; ++j)
            {
                double u, v;
                if (_w <= 0)
                {
                    u = (_x > 0) ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
                    v = (_y > 0) ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
                }
                else
                {
                    double x = _x / _w, y = _y / _w;

                    double r = sqrt(x * x + y * y);
                    double theta = atan(r);

                    double theta2 = theta * theta, theta4 = theta2 * theta2, theta6 = theta4 * theta2, theta8 = theta4 * theta4;
                    double theta_d = theta * (1 + k[0] * theta2 + k[1] * theta4 + k[2] * theta6 + k[3] * theta8);

                    double scale = (r == 0) ? 1.0 : theta_d / r;
                    u = f[0] * x * scale + c[0];
                    v = f[1] * y * scale + c[1];
                }

                if (m_m1type == CV_16SC2)
                {
                    int iu = cv::saturate_cast<int>(u * cv::INTER_TAB_SIZE);
                    int iv = cv::saturate_cast<int>(v * cv::INTER_TAB_SIZE);
                    m1[j * 2 + 0] = (short)(iu >> cv::INTER_BITS);
                    m1[j * 2 + 1] = (short)(iv >> cv::INTER_BITS);
                    m2[j] = (ushort)((iv & (cv::INTER_TAB_SIZE - 1)) * cv::INTER_TAB_SIZE + (iu & (cv::INTER_TAB_SIZE - 1)));
                }
                else if (m_m1type == CV_32FC1)
                {
                    m1f[j] = (float)u;
                    m2f[j] = (float)v;
                }

                _x += iR(0, 0);
                _y += iR(1, 0);
                _w += iR(2, 0);
            }
        }
    }

private:
    cv::Mat& m_map1;
    cv::Mat& m_map2;
    cv::Mat m_K;
    cv::Mat m_D;
    cv::Mat m_R;
    cv::Mat m_P;
    Matx33f camMat;
    cv::Vec2d f, c;
    Vec4d k;
    cv::Matx33d iR;
    const cv::Size& m_size;
    int m_m1type;
};


class FastParallelUndistort : public cv::ParallelLoopBody {
public:
    FastParallelUndistort(cv::Mat& map1, cv::Mat& map2, cv::Mat K, cv::Mat D, cv::Mat R, cv::Mat P, const cv::Size& size, int m1type)
        :m_map1(map1), m_map2(map2), m_K(K), m_D(D), m_R(R), m_P(P), m_size(size), m_m1type(m1type) {

        CV_Assert(m_m1type == CV_16SC2 || m_m1type == CV_32F || m_m1type <= 0);
        m_map1.create(m_size, m1type <= 0 ? CV_16SC2 : m1type);
        m_map2.create(m_size, m_map1.type() == CV_16SC2 ? CV_16UC1 : CV_32F);

        CV_Assert((m_K.depth() == CV_32F || m_K.depth() == CV_64F) && (m_D.depth() == CV_32F || m_D.depth() == CV_64F));
        CV_Assert((m_P.empty() || m_P.depth() == CV_32F || m_P.depth() == CV_64F) && (m_R.empty() || m_R.depth() == CV_32F || m_R.depth() == CV_64F));
        CV_Assert(m_K.size() == cv::Size(3, 3) && (m_D.empty() || m_D.total() == 4));
        CV_Assert(m_R.empty() || m_R.size() == cv::Size(3, 3) || m_R.total() * m_R.channels() == 3);
        CV_Assert(m_P.empty() || m_P.size() == cv::Size(3, 3) || m_P.size() == cv::Size(4, 3));


        if (m_K.depth() == CV_32F)
        {
            camMat = m_K;
            f = Vec2f(camMat(0, 0), camMat(1, 1));
            c = Vec2f(camMat(0, 2), camMat(1, 2));
        }
        else
        {
            camMat = m_K;
            f = Vec2d(camMat(0, 0), camMat(1, 1));
            c = Vec2d(camMat(0, 2), camMat(1, 2));
        }


        k = Vec4d::all(0);
        if (!m_D.empty())
            k = m_D.depth() == CV_32F ? (Vec4d)*m_D.ptr<Vec4f>() : *m_D.ptr<Vec4d>();

        cv::Matx33d RR = cv::Matx33d::eye();
        if (!m_R.empty() && m_R.total() * m_R.channels() == 3)
        {
            cv::Vec3d rvec;
            m_R.convertTo(rvec, CV_32F);
            RR = Affine3d(rvec).rotation();
        }
        else if (!m_R.empty() && m_R.size() == Size(3, 3))
            m_R.convertTo(RR, CV_32F);

        cv::Matx33d PP = cv::Matx33d::eye();
        if (!m_P.empty())
            m_P.colRange(0, 3).convertTo(PP, CV_32F);

        iR = (PP * RR).inv(cv::DECOMP_SVD);

    }

    virtual void operator () (const cv::Range& range) const CV_OVERRIDE {

        for (int i = range.start; i < range.end; ++i)
        {
            float* m1f = m_map1.ptr<float>(i);
            float* m2f = m_map2.ptr<float>(i);
            short* m1 = (short*)m1f;
            ushort* m2 = (ushort*)m2f;

            float _x = i * iR(0, 1) + iR(0, 2),
                _y = i * iR(1, 1) + iR(1, 2),
                _w = i * iR(2, 1) + iR(2, 2);

            for (int j = 0; j < m_size.width; ++j)
            {
                float u, v;
                if (_w <= 0)
                {
                    u = (_x > 0) ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
                    v = (_y > 0) ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
                }
                else
                {
                    float x = _x / _w, y = _y / _w;

                    float r = sqrtf(x * x + y * y);
                    float theta = atanf(r);

                    float theta2 = theta * theta, theta4 = theta2 * theta2, theta6 = theta4 * theta2, theta8 = theta4 * theta4;
                    float theta_d = theta * (1 + k[0] * theta2 + k[1] * theta4 + k[2] * theta6 + k[3] * theta8);

                    float scale = (r == 0) ? 1.0 : theta_d / r;
                    u = f[0] * x * scale + c[0];
                    v = f[1] * y * scale + c[1];
                }

                if (m_m1type == CV_16SC2)
                {
                    int iu = cv::saturate_cast<int>(u * cv::INTER_TAB_SIZE);
                    int iv = cv::saturate_cast<int>(v * cv::INTER_TAB_SIZE);
                    m1[j * 2 + 0] = (short)(iu >> cv::INTER_BITS);
                    m1[j * 2 + 1] = (short)(iv >> cv::INTER_BITS);
                    m2[j] = (ushort)((iv & (cv::INTER_TAB_SIZE - 1)) * cv::INTER_TAB_SIZE + (iu & (cv::INTER_TAB_SIZE - 1)));
                }
                else if (m_m1type == CV_32FC1)
                {
                    m1f[j] = (float)u;
                    m2f[j] = (float)v;
                }

                _x += iR(0, 0);
                _y += iR(1, 0);
                _w += iR(2, 0);
            }
        }
    }

private:
    cv::Mat& m_map1;
    cv::Mat& m_map2;
    cv::Mat m_K;
    cv::Mat m_D;
    cv::Mat m_R;
    cv::Mat m_P;
    Matx33f camMat;
    cv::Vec2d f, c;
    Vec4d k;
    cv::Matx33d iR;
    const cv::Size& m_size;
    int m_m1type;
};

using namespace std;


int main() {

    std::string videofile = "C:/Users/elvin/Documents/code/gyroflow/test_clips/IF-RC01_0011.MP4";
    std::ifstream infile("C:/Users/elvin/Documents/code/gyroflow/test_clips/IF-RC01_0011.MP4.gyroflow");

    cv::VideoCapture cap(videofile);
    if (!cap.isOpened()) {
        return -1;
    }

    //std::cout << rotMatFromQuat(1, 2, 3, 4);
    //return 0;

    json gyroflowData;

    infile >> gyroflowData;

    //std::cout << gyroflow_data["raw_imu"][100];

    //if (gyroflow_data["stab_transform"][100].is_null()) {
    //	std::cout << "it's a null\n";
    //}

    std::string line;
    std::int32_t frame_num;
    /*
    while (std::getline(infile, line) && counter < 100) {
        //std::istringstream iss(line);
        std::cout << line;
        counter++;
    }*/

    json cam_params = gyroflowData["calibration_data"]["fisheye_params"];
    cv::Mat Kt(3, 3, CV_32FC1);
    cv::Mat m_P(3, 3, CV_32FC1);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Kt.at<float>(i, j) = cam_params["camera_matrix"][i][j];
            m_P.at<float>(i, j) = cam_params["camera_matrix"][i][j];
        }
    }

    cv::Size out_size;
    out_size.width = 1920;
    out_size.height = 1440;

    cv::Size in_sizet;
    in_sizet.width = gyroflowData["video_info"]["orig_w"];
    in_sizet.height = gyroflowData["video_info"]["orig_h"];

    float im_dim_ratio = out_size.height / in_sizet.height;

    Kt.at<float>(0, 0) *= im_dim_ratio;
    Kt.at<float>(0, 2) *= im_dim_ratio;
    Kt.at<float>(1, 1) *= im_dim_ratio;
    Kt.at<float>(1, 2) *= im_dim_ratio;

    m_P.at<float>(0, 0) /= 2.5;
    m_P.at<float>(1, 1) /= 2.5;
    m_P.at<float>(0, 2) = out_size.width/2;
    m_P.at<float>(1, 2) = out_size.height / 2;

    //std::cout << K;

    cv::Mat m_D(4, 1, CV_32FC1);
    for (int i = 0; i < 4; i++) {
        m_D.at<float>(i) = cam_params["distortion_coeffs"][i];
    }
    cv::Mat frame;
    cv::Mat frame_out;
    cv::Mat map1;
    cv::Mat map2;

    map1.create(out_size, CV_32F);
    map2.create(out_size, CV_32F);

    cv::Size size;
    size.width = gyroflowData["video_info"]["orig_w"];
    size.height = gyroflowData["video_info"]["orig_h"];
    


    cv::Size size_out(960, 720);

    //cap.set(cv::CAP_PROP_POS_MSEC, 60000);

    clock_t begin_time = std::clock();
    for (int fr = 0; fr< 60; fr++) {





        //readGyroflowFile();

        //return 0;

        //
        /*try {*/
        frame_num = cap.get(cv::CAP_PROP_POS_FRAMES);
        cap >> frame;

        if (gyroflowData["stab_transform"][frame_num].is_null()) {
            break;
        }

        json quat = gyroflowData["stab_transform"][frame_num];

        cv::Mat m_R;
        if (quat.is_null()) {
            m_R = stabRotMatFromQuat(0, 1, 0, 0);
        }
        else {
            m_R = stabRotMatFromQuat(quat[3], quat[4], quat[5], quat[6]);
            //m_R = stabRotMatFromQuat(0.924, 0.383, 0.1, 0);
        }

        /*
        cv::Vec2d f, c;
        Vec4d k;
        cv::Matx33d iR;
        cv::Matx33f camMat;

        if (Kt.depth() == CV_32F)
        {
            camMat = Kt;
            f = Vec2f(camMat(0, 0), camMat(1, 1));
            c = Vec2f(camMat(0, 2), camMat(1, 2));
        }
        else
        {
            camMat = Kt;
            f = Vec2d(camMat(0, 0), camMat(1, 1));
            c = Vec2d(camMat(0, 2), camMat(1, 2));
        }


        k = Vec4d::all(0);
        if (!m_D.empty())
            k = m_D.depth() == CV_32F ? (Vec4d)*m_D.ptr<Vec4f>() : *m_D.ptr<Vec4d>();


        cv::Matx33d RR = cv::Matx33d::eye();
        if (!m_R.empty() && m_R.total() * m_R.channels() == 3)
        {
            cv::Vec3d rvec;
            m_R.convertTo(rvec, CV_64F);
            RR = Affine3d(rvec).rotation();
        }
        else if (!m_R.empty() && m_R.size() == Size(3, 3))
            m_R.convertTo(RR, CV_64F);

        cv::Matx33d PP = cv::Matx33d::eye();
        if (!m_P.empty())
            m_P.colRange(0, 3).convertTo(PP, CV_64F);

        iR = (PP * RR).inv(cv::DECOMP_SVD);
        */

        ParallelUndistort parallelUndistort(map1, map2, Kt, m_D, m_R, m_P, size, CV_32FC1);
        cv::parallel_for_(cv::Range(0, size.height), parallelUndistort);
        //std::cout << (int)map1.at<float>(100, 100) << "," << (int)map2.at<float>(100, 100) << "\n";

        //cv::Size in_sizet(2704, 2028);

        //cv::fisheye::initUndistortRectifyMap(Kt, Dt, stabRotMatFromQuat(1, 0, 0, 0), Kt, size, CV_32FC1, map1t, map2t);

        /*
        for (int _i = 0; _i < out_size.height; _i++) {

            //std::cout << procWindow.y1 << "," << procWindow.y2 << "\n";

            //int i = out_size.height - 1 - _i;
            int i = _i;


            double _x = i * iR(0, 1) + iR(0, 2),
                _y = i * iR(1, 1) + iR(1, 2),
                _w = i * iR(2, 1) + iR(2, 2);

            for (int j = 0; j < out_size.width; j++) {


                double u, v;
                if (_w <= 0)
                {
                    u = (_x > 0) ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
                    v = (_y > 0) ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
                }
                else
                {
                    double x = _x / _w, y = _y / _w;

                    double r = sqrt(x * x + y * y);
                    double theta = atan(r);

                    double theta2 = theta * theta, theta4 = theta2 * theta2, theta6 = theta4 * theta2, theta8 = theta4 * theta4;
                    double theta_d = theta * (1 + k[0] * theta2 + k[1] * theta4 + k[2] * theta6 + k[3] * theta8);

                    double scale = (r == 0) ? 1.0 : theta_d / r;
                    u = f[0] * x * scale + c[0];
                    v = f[1] * y * scale + c[1];
                }


                //m1f[j] = (float)u;
                //m2f[j] = (float)v;


                _x += iR(0, 0);
                _y += iR(1, 0);
                _w += iR(2, 0);


                //int nx = x +  (int) (std::sin((double)x / 10) * 15);
                //int ny = procWindow.y2 - 1 - (int)v;
                int ny = (int)v;
                int nx = (int)u;

                map1.at<float>(i, j) = nx;
                map2.at<float>(i, j) = ny;

            }
        }*/


        //cv::fisheye::initUndistortRectifyMap(Kt, m_D, m_R, m_P, out_size, CV_32FC1, map1, map2);
        cv::remap(frame, frame_out, map1, map2, cv::INTER_LINEAR);
        //cv::resize(frame_out, frame_out, size_out);

        //cv::imshow("Frame", frame_out);

    /*
    }
    catch (const std::exception& e) {
    }*/



        if (cv::waitKey(1) >= 0) {
            break;
        }

    }

    
    std::cout << "Standard fps: " << 100.0 / (float(clock() - begin_time) / CLOCKS_PER_SEC) << "\n";


    begin_time = std::clock();
    for (int c = 0; c < 60; c++) {

        //readGyroflowFile();

        //return 0;

        //
        /*try {*/
        frame_num = cap.get(cv::CAP_PROP_POS_FRAMES);
        cap >> frame;

        if (gyroflowData["stab_transform"][frame_num].is_null()) {
            break;
        }

        json quat = gyroflowData["stab_transform"][frame_num];

        cv::Mat m_R;
        if (quat.is_null()) {
            m_R = stabRotMatFromQuat(0, 1, 0, 0);
        }
        else {
            m_R = stabRotMatFromQuat(quat[3], quat[4], quat[5], quat[6]);
            //m_R = stabRotMatFromQuat(0.924, 0.383, 0.1, 0);
        }

        FastParallelUndistort parallelUndistort(map1, map2, Kt, m_D, m_R, m_P, size, CV_32FC1);
        cv::parallel_for_(cv::Range(0, size.height), parallelUndistort);

        
        //cv::fisheye::initUndistortRectifyMap(Kt,m_D, m_R, m_P,size, CV_32FC1,map1,map2);
        cv::remap(frame, frame_out, map1, map2, cv::INTER_LINEAR);
        //cv::resize(frame_out, frame_out, size_out);

        //cv::imshow("Frame", frame_out);

        /*
        }
        catch (const std::exception& e) {
        }*/



        if (cv::waitKey(1) >= 0) {
            break;
        }

    }

    std::cout << "Parallel fps: " << 100.0 / (float(clock() - begin_time) / CLOCKS_PER_SEC);

    return 0;
}

cv::Mat stabRotMatFromQuat(float w, float i, float j, float k) {
    cv::Mat out(3, 3, CV_32FC1);
    out.at<float>(0, 0) = 1 - 2 * (j * j + k * k);
    out.at<float>(0, 1) = -2 * (i * j - k * w);
    out.at<float>(0, 2) = -2 * (i * k + j * w);
    out.at<float>(1, 0) = -2 * (i * j + k * w);
    out.at<float>(1, 1) = 1 - 2 * (i * i + k * k);
    out.at<float>(1, 2) = 2 * (j * k - i * w);
    out.at<float>(2, 0) = -2 * (i * k - j * w);
    out.at<float>(2, 1) = 2 * (j * k + i * w);
    out.at<float>(2, 2) = 1 - 2 * (i * i + j * j);

    return out;
}