    //Includes all the headers necessary to use the most common public pieces of the ROS system.
    #include <ros/ros.h>
    //Use image_transport for publishing and subscribing to images in ROS
    #include <image_transport/image_transport.h>
    //Use cv_bridge to convert between ROS and OpenCV Image formats
    #include <cv_bridge/cv_bridge.h>

    #include <sensor_msgs/image_encodings.h>
    //Include headers for OpenCV Image processing
    #include <opencv2/imgproc/imgproc.hpp>
    //Include headers for OpenCV GUI handling
    #include <opencv2/highgui/highgui.hpp>
    #include<string>
    #include <sstream>
    using namespace cv;
    using namespace std;

    //Store all constants for image encodings in the enc namespace to be used later.
    namespace enc = sensor_msgs::image_encodings;
    void image_socket(Mat inImg);
    /*
    Mat是OpenCV最基本的数据结构，Mat即矩阵（Matrix）的缩写，Mat数据结构主要包含2部分：Header和Pointer。
    Header中主要包含矩阵的大小，存储方式，存储地址等信息；
    Pointer中存储指向像素值的指针。我们在读取图片的时候就是将图片定义为Mat类型，其重载的构造函数一大堆，
    */
    Mat image1;
    static int imgWidth, imgHeight;

    //char *output_file = "/home/exbot/kinect_picture/";

    //This function is called everytime a new image_info message is published图像信息高和宽
    void camInfoCallback(const sensor_msgs::CameraInfo & camInfoMsg)
    {
      //Store the image width for calculation of angle
      imgWidth = camInfoMsg.width;
      imgHeight = camInfoMsg.height;
    // ROS_INFO("%ld,%ld",imgWidth/6,imgHeight/6);
    }

    //This function is called everytime a new image is published有新图发布就调用
    void imageCallback(const sensor_msgs::ImageConstPtr& original_image)
    {
        //Convert from the ROS image message to a CvImage suitable for working with OpenCV for processing
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            //Always copy, returning a mutable CvImage返回一个RGB图像
            //OpenCV expects color images to use BGR channel order.
            cv_ptr = cv_bridge::toCvCopy(original_image, enc::BGR8);
        }
        catch (cv_bridge::Exception& e)
        {
            //if there is an error during conversion, display it
            ROS_ERROR("tutorialROSOpenCV::main.cpp::cv_bridge exception: %s", e.what());
            return;
        }
        image_socket(cv_ptr->image);

    }

    void image_socket(Mat inImg)
    {
       imshow("image_socket", inImg);//显示图片
        if( inImg.empty() )
        {
          ROS_INFO("Camera image empty");
          return;//break;
        }
        stringstream sss;
        string strs;
        static int image_num = 1;
        char c = (char)waitKey(1);

        if( c == 27 )
          ROS_INFO("Exit boss");//break;
        switch(c)
        {
          case 'p':
          resize(inImg,image1,Size(imgWidth/6,imgHeight/6),0,0,CV_INTER_LINEAR);
          image1=image1(Rect(image1.cols/2-32,image1.rows/2-32, image1.cols/2+32, image1.rows/2+32));

          strs="/home/exbot/kinect_picture/";
          sss.clear();
          sss<<strs;
          sss<<image_num;
          sss<<".jpg";
          sss>>strs;
          imwrite(strs,image1);//保存图片
          image_num++;
          break;
      default:
          break;
      }

    }


    /**
    * This is ROS node to track the destination image
    */
    int main(int argc, char **argv)
    {
        ros::init(argc, argv, "image_socket");
        ROS_INFO("-----------------");

        ros::NodeHandle nh;

        image_transport::ImageTransport it(nh);

        image_transport::Subscriber sub = it.subscribe("/camera/rgb/image_raw", 1, imageCallback);
        ros::Subscriber camInfo         = nh.subscribe("/camera/rgb/camera_info", 1, camInfoCallback);

        ros::spin();

        //ROS_INFO is the replacement for printf/cout.
        ROS_INFO("tutorialROSOpenCV::main.cpp::No error.");
    }
