#include "StudentTasks.hpp"
#include <opencv2/highgui.hpp>

int main()
{
    cv::Mat frame;
    while (true)
    {
        if (!get_pic(frame))
            continue;
        auto dets = armor_detect(frame, TeamColor::Red);
        drawDetections(frame, dets);
        cv::imshow("live", frame);
        if (cv::waitKey(1) == 27)
            break;
    }
}