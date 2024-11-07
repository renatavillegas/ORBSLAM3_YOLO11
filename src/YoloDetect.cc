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
		mModule = torch::jit::load("yolo11n-seg.torchscript");
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
	void YoloDetect::AddNewObject(int area_x, int area_y, int area_width, int area_height, std::string classID, cv::Mat objectMask, std::pair<float, float> depthMinMax)
	{
		std::lock_guard<std::mutex> lock(mMutex);
		Object newObject;
		newObject.area = cv::Rect2i(area_x, area_y, area_width, area_height);
		cout<<"newObjectArea="<< newObject.area<<endl;
		newObject.classID = classID;
		newObject.mapPoints = vector<MapPoint*>(1,static_cast<MapPoint*>(NULL));
		newObject.objectMask = objectMask;
		newObject.depthMinMax = depthMinMax;
		mObjects.push_back(newObject);			
	}
	void YoloDetect::ClearObjects()
	{
		mObjects.clear();
	}
	std::pair<float, float> YoloDetect::CalculateDepth(cv::Rect boudingboxLeft, cv::Rect boudingboxRight, float bf)
	{
		//calulate disparity min and max
		int minXLeft, maxXLeft, minXRight, maxXRight;
		minXLeft = boudingboxLeft.x; 
		maxXLeft = boudingboxLeft.x + boudingboxLeft.width;
		minXRight = boudingboxRight.x;
		maxXRight = boudingboxRight.x + boudingboxRight.width;

		float disparityMin = static_cast<float>(minXLeft - minXRight);
		float disparityMax = static_cast<float>(maxXLeft - maxXRight);
		
		if ((disparityMin <= 0)||disparityMax <=0){
			 std::cerr << "Disparity must be greater than 0!" << std::endl;
			 return std::make_pair(-1,-1);
		}
		return std::make_pair(bf/disparityMax, bf/disparityMin);

	}
	void YoloDetect::Detect()
	{
		cv::Mat imgLeft, imgRight;
		//std::lock_guard<std::mutex> lock(mMutex);
        if(mImageLeft.empty()||mImageRight.empty())
         	return;
	    // Preparing input tensor left
	    cv::resize(mImageLeft, imgLeft, cv::Size(640, 640));
	    cv::cvtColor(imgLeft, imgLeft, cv::COLOR_BGR2RGB);
	    torch::Tensor imgTensorLeft = torch::from_blob(imgLeft.data, {imgLeft.rows, imgLeft.cols,3},torch::kByte);
	    imgTensorLeft = imgTensorLeft.permute({2, 0, 1}); //channels, height, width
	    imgTensorLeft = imgTensorLeft.unsqueeze(0);
	    imgTensorLeft = imgTensorLeft.toType(torch::kFloat);
	    imgTensorLeft = imgTensorLeft.div(255.0);
	    //execute inference
	    std::vector<torch::jit::IValue> inputs;
	    inputs.push_back(std::move(imgTensorLeft));
	    torch::jit::IValue output = mModule.forward(inputs);
	    //extract predictions
	    auto preds = output.toTuple()->elements();
	    cout << "preds.size=" <<preds.size()<< endl;

		for (size_t i = 0; i < preds.size(); ++i) {
		    torch::Tensor tensor = preds[i].toTensor();
		    cout << "Tensor " << i << ": " << tensor.sizes() << std::endl;
		}
		torch::Tensor detections = preds[0].toTensor();
		detections = detections.transpose(1, 2).contiguous();

		vector<torch::Tensor> det = non_max_suppression_seg(detections, 0.5, 0.5);
		cout << "det.size =" << det.size();
	    // Prepare segmentation mask.
	    //similar github https://github.com/kimwoonggon/Cpp_Libtorch_DLL_YoloV8Segmentation_CSharpProject/blob/7fd1386da091fd4c7382ef258c3ac8077af5bbb8/YoloV8DLLProject/dllmain.cpp#L381

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
        //cout << "scores: " << scores << endl;
        pred = torch::index_select(pred, 0, torch::nonzero(scores > score_thresh).select(1, 0));
//        cout << "preds " << i << "=" << pred<<endl;
        //until here it's right. 

    //     if (pred.sizes()[0] == 0) continue;

         // (center_x, center_y, w, h) to (left, top, right, bottom)
         // pred.select(1, 0) = pred.select(1, 0) - pred.select(1, 2) / 2;
         // pred.select(1, 1) = pred.select(1, 1) - pred.select(1, 3) / 2;
         // pred.select(1, 2) = pred.select(1, 0) + pred.select(1, 2);
         // pred.select(1, 3) = pred.select(1, 1) + pred.select(1, 3);
         //cout<< "pred="<< pred << endl;
//		 output.push_back(pred);
//	}
        // Computing scores and classes
        std::tuple<torch::Tensor, torch::Tensor> max_tuple = torch::max(pred.slice(1, 5, pred.sizes()[1]), 1);
//        cout << "pred slice: " << pred.slice(1, 5, pred.sizes()[1]) << endl;
//        cout << "pred.sizes(): " << pred.sizes() << endl;
//        cout <<"max_tuple= "<< std::get<0>(max_tuple) <<", " << std::get<1>(max_tuple) << endl;
        pred.select(1, 4) = pred.select(1, 4) * std::get<0>(max_tuple);
        //pred.select(1, 5) = std::get<1>(max_tuple);
//        cout <<"pred.select(1, 4), pred.select(1, 5)= "<< pred.select(1, 4) <<", " << pred.select(1, 5) << endl;

        torch::Tensor  dets = pred.slice(1, 0, 6);

        torch::Tensor keep = torch::empty({dets.sizes()[0]});
        torch::Tensor areas = (dets.select(1, 3) - dets.select(1, 1)) * (dets.select(1, 2) - dets.select(1, 0));
        std::tuple<torch::Tensor, torch::Tensor> indexes_tuple = torch::sort(dets.select(1, 4), 0, 1);
        torch::Tensor v = std::get<0>(indexes_tuple);
        torch::Tensor indexes = std::get<1>(indexes_tuple);
//        cout << "v=" << v <<endl;
//        cout << "indexes" << indexes<<endl;
        int count = 0;
        while (indexes.sizes()[0] > 0)
        {
            keep[count] = (indexes[0].item().toInt());
            count += 1;

    //         // Computing overlaps
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

    //         // FIlter by IOUs
            torch::Tensor ious = overlaps / (areas.select(0, indexes[0].item().toInt()) + torch::index_select(areas, 0, indexes.slice(0, 1, indexes.sizes()[0])) - overlaps);
            indexes = torch::index_select(indexes, 0, torch::nonzero(ious <= iou_thresh).select(1, 0) + 1);
        }
        keep = keep.toType(torch::kInt64);
        output.push_back(torch::index_select(dets, 0, keep.slice(0, 0, count)));
//        cout << "keep="<< keep <<endl;
//        cout << "count=" << count <<endl;
        cout << "output="<< output<< endl;;
    }

    return output;
}

vector<torch::Tensor> YoloDetect::non_max_suppression_seg(torch::Tensor preds, float score_thresh, float iou_thresh)
{
	std::vector<torch::Tensor> output;
	for (size_t i = 0; i < preds.sizes()[0]; ++i) {
		 torch::Tensor pred = preds.select(0, i);
		 torch::Tensor scores = std::get<0>(torch::max(pred.slice(1, 4, 84), 1));
		 auto mask = scores > score_thresh;
		 if (mask.sum().item<int>() > 0) {
		 	torch::Tensor indices = torch::nonzero(mask).select(1, 0);
		 	pred = torch::index_select(pred, 0, indices);
		 	output.push_back(pred);
		 } else {
		 	continue;
		 }
		 //pred = torch::index_select(pred, 0, torch::nonzero(scores > score_thresh).select(1, 0));
		 //cout <<"scores="<< scores << endl;
	}
	return output;
}

    void YoloDetect::GetImage(cv::Mat &imageLeft, cv::Mat &imageRight)
	{
		std::lock_guard<std::mutex> lock(mMutex);
    	mImageLeft = imageLeft;
    	mImageRight = imageRight;
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
