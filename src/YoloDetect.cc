#include "MapPoint.h"
#include <opencv2/opencv.hpp>
#include "YoloDetect.h"


namespace ORB_SLAM3
{
	static const int INPUT_W = 640;
	static const int INPUT_H = 640;
	YoloDetect::YoloDetect()
	{
		LoadClassNames();
	}
void YoloDetect::LoadClassNames()
	{
		//load model 
		mModule = torch::jit::load("yolov10n.torchscript");
		//load classes
		std::ifstream f("/home/oficina-robotica/ORB_SLAM3_2/coco.names");
	    if (!f.is_open())
	    {
	        std::cerr << "Error: Could not open file coco.names" << std::endl;
	        return; // return early if file cannot be opened
	    }
    	std::string name = "";
   	 	while (std::getline(f, name))
     	{
     		std::lock_guard<std::mutex> lock(mMutex); 
        	mClassnames.push_back(name);
    	}
	}
	//Just for test
	void YoloDetect::AddNewObject(int area_x, int area_y, int area_width, int area_height, std::string classID, cv::Mat objectMask)
	{
		std::lock_guard<std::mutex> lock(mMutex);
		Object newObject;
		newObject.area = cv::Rect2i(area_x, area_y, area_width, area_height);
		cout<<"newObjectArea="<< newObject.area<<endl;
		newObject.classID = classID;
		newObject.mapPoints = vector<MapPoint*>(1,static_cast<MapPoint*>(NULL));
		newObject.objectMask = objectMask;
		mObjects.push_back(newObject);			
	}
	void YoloDetect::ClearObjects()
	{
		mObjects.clear();
	}
	void YoloDetect::Detect()
	{
		cv::Mat img;
		//std::lock_guard<std::mutex> lock(mMutex);
        if(mImage.empty())
         	return;
	    // Preparing input tensor
	    cv::resize(mImage, img, cv::Size(640, 640));
	    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
	    torch::Tensor imgTensor = torch::from_blob(img.data, {img.rows, img.cols,3},torch::kByte);
	    imgTensor = imgTensor.permute({2,0,1});
	    imgTensor = imgTensor.toType(torch::kFloat);
	    imgTensor = imgTensor.div(255);
	    imgTensor = imgTensor.unsqueeze(0);	
	    torch::Tensor preds =  mModule.forward({imgTensor}).toTensor().cpu();
    	std::vector<torch::Tensor> dets = YoloDetect::non_max_suppression(preds, 0.8, 0.5);
    	//binary mask.
    	cv::Mat objectMask = cv::Mat::zeros(mImage.size(), CV_8UC1) * 255;
    	if (dets.size() > 0)
    	{
//    		cout << "dets.size()="<<dets.size()<< " dets[0].sizes()[0]="<<dets[0].sizes()[0] << endl; 
    		if(dets[0].sizes()[0]>mObjects.size())
    		{
	    		int x, y, l, h, left, top, bottom, right, index;
	    		std::string classID = "";
		        // Visualize result
		        for (size_t i=0; i < dets[0].sizes()[0]; ++ i)
		        {
		            left = dets[0][i][0].item().toFloat() * mImage.cols / 640;
		            top = dets[0][i][1].item().toFloat() * mImage.rows / 640;
		            right = dets[0][i][2].item().toFloat() * mImage.cols / 640;
		            bottom = dets[0][i][3].item().toFloat() * mImage.rows / 640;
		            index = dets[0][i][5].item().toInt();
		            classID = mClassnames[index];
	                left = std::max(0, left);
	                top = std::max(0, top);
	                right = std::min(mImage.cols, right);
	                bottom = std::min(mImage.rows, bottom);		            
			        x = std::max(0, left);
			        y = bottom;
			        l = right - left; 
			        h = bottom-top;		            
		            cv::rectangle(mImage, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 255, 0), 2);
		            cv::putText(mImage, classID, cv::Point(left, top - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
					cv::Rect objectROI(left, top, l, h);
					//cout<<"Rec="<< objectROI<< endl;
					objectMask(objectROI).setTo(cv::Scalar(1));
		            //objectMask= mImage(objectROI).clone();
		           	//cv::imshow("Detections", objectMask);
		           	//cout<<"Rec="<< objectROI<< endl << "image size = "<< mImage.rows << "x" <<mImage.cols <<endl;
		            //cout << "x="<<x<<",y="<<y<<", l="<<l<<", h="<<h <<endl;
		            cout<<"classID="<<classID<<endl;
		        	AddNewObject(x,y,l,h, classID, objectMask);

		        }
		        cv::waitKey(10);
    		}
	    }
		return;
	}

	std::vector<YoloDetect::Object> YoloDetect::GetObjects()
    {
    	if(!mObjects.empty())
        	std::lock_guard<std::mutex> lock(mMutex); 
        	return mObjects;
        return mObjects;
    }
    void YoloDetect::SetMapPoints(int objectIndex, const std::vector<MapPoint*>& newMapPoints)
    {
    	std::lock_guard<std::mutex> lock(mMutex);
    	if (objectIndex >= 0 && objectIndex < mObjects.size())
	    {
	        mObjects[objectIndex].mapPoints = newMapPoints; // Set new mapPoints for the specified object
	    }
    }
    void YoloDetect::SetKeyPoints(int objectIndex, const std::vector<cv::KeyPoint>& newKeyPoints)
    {
        std::lock_guard<std::mutex> lock(mMutex);
	    if (objectIndex >= 0 && objectIndex < mObjects.size())
	    {
	        mObjects[objectIndex].keyPoints = newKeyPoints; // Set new keyPoints for the specified object.
	    }
    }

vector<torch::Tensor> YoloDetect::non_max_suppression(torch::Tensor preds, float score_thresh, float iou_thresh)
{
    std::vector<torch::Tensor> output;
    for (size_t i=0; i < preds.sizes()[0]; ++i)
    {
        torch::Tensor pred = preds.select(0, i);

        // Filter by scores
        torch::Tensor scores = pred.select(1, 4);
        pred = torch::index_select(pred, 0, torch::nonzero(scores > score_thresh).select(1, 0));
        if (pred.sizes()[0] == 0) continue;

        // (center_x, center_y, w, h) to (left, top, right, bottom)
        pred.select(1, 0) = pred.select(1, 0) - pred.select(1, 2) / 2;
        pred.select(1, 1) = pred.select(1, 1) - pred.select(1, 3) / 2;
        pred.select(1, 2) = pred.select(1, 0) + pred.select(1, 2);
        pred.select(1, 3) = pred.select(1, 1) + pred.select(1, 3);

        // Computing scores and classes
        std::tuple<torch::Tensor, torch::Tensor> max_tuple = torch::max(pred.slice(1, 5, pred.sizes()[1]), 1);
        pred.select(1, 4) = pred.select(1, 4) * std::get<0>(max_tuple);
        pred.select(1, 5) = std::get<1>(max_tuple);

        torch::Tensor  dets = pred.slice(1, 0, 6);

        torch::Tensor keep = torch::empty({dets.sizes()[0]});
        torch::Tensor areas = (dets.select(1, 3) - dets.select(1, 1)) * (dets.select(1, 2) - dets.select(1, 0));
        std::tuple<torch::Tensor, torch::Tensor> indexes_tuple = torch::sort(dets.select(1, 4), 0, 1);
        torch::Tensor v = std::get<0>(indexes_tuple);
        torch::Tensor indexes = std::get<1>(indexes_tuple);
        int count = 0;
        while (indexes.sizes()[0] > 0)
        {
            keep[count] = (indexes[0].item().toInt());
            count += 1;

            // Computing overlaps
            torch::Tensor lefts = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor tops = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor rights = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor bottoms = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor widths = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor heights = torch::empty(indexes.sizes()[0] - 1);
            for (size_t i=0; i<indexes.sizes()[0] - 1; ++i)
            {
                lefts[i] = std::max(dets[indexes[0]][0].item().toFloat(), dets[indexes[i + 1]][0].item().toFloat());
                tops[i] = std::max(dets[indexes[0]][1].item().toFloat(), dets[indexes[i + 1]][1].item().toFloat());
                rights[i] = std::min(dets[indexes[0]][2].item().toFloat(), dets[indexes[i + 1]][2].item().toFloat());
                bottoms[i] = std::min(dets[indexes[0]][3].item().toFloat(), dets[indexes[i + 1]][3].item().toFloat());
                widths[i] = std::max(float(0), rights[i].item().toFloat() - lefts[i].item().toFloat());
                heights[i] = std::max(float(0), bottoms[i].item().toFloat() - tops[i].item().toFloat());
            }
            torch::Tensor overlaps = widths * heights;

            // FIlter by IOUs
            torch::Tensor ious = overlaps / (areas.select(0, indexes[0].item().toInt()) + torch::index_select(areas, 0, indexes.slice(0, 1, indexes.sizes()[0])) - overlaps);
            indexes = torch::index_select(indexes, 0, torch::nonzero(ious <= iou_thresh).select(1, 0) + 1);
        }
        keep = keep.toType(torch::kInt64);
        output.push_back(torch::index_select(dets, 0, keep.slice(0, 0, count)));
    }
    return output;
}

    void YoloDetect::GetImage(cv::Mat &image)
	{
		std::lock_guard<std::mutex> lock(mMutex);
    	mImage = image;
	}
	void YoloDetect::Run()
	{
		while(true)
		{
			Detect();
			usleep(10000);
		}
	}
} //namespace ORB_SLAM3
