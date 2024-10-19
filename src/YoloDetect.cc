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
	void YoloDetect::AddNewObject(int area_x, int area_y, int area_width, int area_height, std::string classID, cv::Mat objectMask, float depth)
	{
		std::lock_guard<std::mutex> lock(mMutex);
		Object newObject;
		newObject.area = cv::Rect2i(area_x, area_y, area_width, area_height);
		cout<<"newObjectArea="<< newObject.area<<endl;
		newObject.classID = classID;
		newObject.mapPoints = vector<MapPoint*>(1,static_cast<MapPoint*>(NULL));
		newObject.objectMask = objectMask;
		newObject.object_depth = depth;
		mObjects.push_back(newObject);			
	}
	void YoloDetect::ClearObjects()
	{
		mObjects.clear();
	}
	float YoloDetect::CalculateDepth(int xLeft, int xRight, float bf)
	{
		int disparity = xLeft - xRight;
		if (disparity <= 0){
			 std::cerr << "Disparity must be greater than 0!" << std::endl;
			 return -1.0f;
		}
		return bf / disparity;

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
	    imgTensorLeft = imgTensorLeft.permute({2,0,1});
	    imgTensorLeft = imgTensorLeft.toType(torch::kFloat);
	    imgTensorLeft = imgTensorLeft.div(255);
	    imgTensorLeft = imgTensorLeft.unsqueeze(0);
	    // Preparing input tensor right
	    cv::resize(mImageRight, imgRight, cv::Size(640, 640));
	    cv::cvtColor(imgRight, imgRight, cv::COLOR_BGR2RGB);
	    torch::Tensor imgTensorRight = torch::from_blob(imgRight.data, {imgRight.rows, imgRight.cols, 3 }, torch::kByte);
	    imgTensorRight = imgTensorRight.permute({ 2, 0, 1 });
	    imgTensorRight = imgTensorRight.toType(torch::kFloat);
	    imgTensorRight = imgTensorRight.div(255);
	    imgTensorRight = imgTensorRight.unsqueeze(0);

	    //dets on left side 
	    torch::Tensor predsLeft =  mModule.forward({imgTensorLeft}).toTensor().cpu();
    	std::vector<torch::Tensor> detsLeft = YoloDetect::non_max_suppression(predsLeft, 0.3, 0.3);
    	//dets on right side 
    	torch::Tensor predsRight = mModule.forward({ imgTensorRight }).toTensor().cpu();
    	std::vector<torch::Tensor> detsRight = YoloDetect::non_max_suppression(predsRight, 0.3, 0.3);
    	//cout << "detsRight size, detsLeft size = " << detsRight.size() <<", " << detsLeft.size()<<endl;
    	if (!detsLeft.empty() && !detsRight.empty())
    	{
    		//cout << "dets.size()="<<dets.size()<< " dets[0].sizes()[0]="<<dets[0].sizes()[0] << endl; 
    		if(detsLeft[0].sizes()[0]>mObjects.size())
    		{
	    		int x_left, y_left, l_left, h_left, imleft_left, imleft_top, imleft_bottom, imleft_right, imleft_index;
	    		int x_right, y_right, l_right, h_right, imright_left, imright_top, imright_bottom, imright_right, imright_index;
	    		std::string classID_left = "";
	    		std::string classID_right = "";
	    		//store the center of the bounding box from the left and right images. 
	    		std::vector<cv::Point2f> centersLeft, centersRight;
		        // Image Left Processing
		        for (size_t i=0; i < detsLeft[0].sizes()[0]; ++ i)
		        {
		        	//binary mask.
    				cv::Mat objectMask = cv::Mat::zeros(mImageLeft.size(), CV_8UC1);
		            imleft_left =  std::max(0, (int)detsLeft[0][i][0].item().toFloat() * mImageLeft.cols / 640);
		            imleft_top = std::max(0, (int)detsLeft[0][i][1].item().toFloat() * mImageLeft.rows / 640);
		            imleft_right = std::min(mImageLeft.cols, (int)detsLeft[0][i][2].item().toFloat() * mImageLeft.cols / 640);
		            imleft_bottom = std::min(mImageLeft.rows, (int)detsLeft[0][i][3].item().toFloat() * mImageLeft.rows / 640);
		            imleft_index = detsLeft[0][i][5].item().toInt();
		            classID_left = mClassnames[imleft_index];
			        l_left = imleft_right - imleft_left;
			        h_left = imleft_bottom-imleft_top;       
					cv::Rect leftBox(imleft_left, imleft_top, l_left, h_left);
					centersLeft.push_back(cv::Point2f((imleft_left + imleft_right) / 2.0, (imleft_top + imleft_bottom) / 2.0));
					cout<<"centersLeft="<< centersLeft[i]<< endl;
					
		           	//cout<<"Rec="<< objectROI<< endl << "image size = "<< mImageLeft.rows << "x" <<mImageLeft.cols <<endl;
		            //cout << "x="<<x<<",y="<<y<<", l="<<l<<", h="<<h <<endl;
		            //cout<<"classID="<<classID<<endl;

		        	float bestMatchScore = std::numeric_limits<float>::max();
		        	int bestMatchIndex = -1;

		        	//check for correspondece on the right image
				    for (size_t j = 0; j < detsRight[0].sizes()[0]; ++j)
				    {
				        imright_index = detsRight[0][j][5].item().toInt();
				        classID_right=mClassnames[imright_index];
				        if(classID_left!=classID_right)
				        	continue;
				        imright_left = std::max(0,(int)detsRight[0][j][0].item().toFloat() * mImageRight.cols / 640);
				        imright_top = std::max(0, (int)detsRight[0][j][1].item().toFloat() * mImageRight.rows / 640);
				        imright_right = std::min(mImageRight.cols, (int)detsRight[0][j][2].item().toFloat() * mImageRight.cols / 640);
				        imright_bottom = std::min(mImageRight.rows, (int)detsRight[0][j][3].item().toFloat() * mImageRight.rows / 640);
				        l_right= imright_right - imright_left;
				        h_right= imright_bottom-imright_top;
				        cv::Rect rightBox(imright_left, imright_top, l_right,h_right);
				        centersRight.push_back(cv::Point2f((imright_left + imright_right) / 2.0, (imright_top + imright_bottom) / 2.0));
				        float disparity = std::abs(imleft_left - imright_left);
			            if (disparity > 0 && disparity < bestMatchScore)
			            {
			                bestMatchScore = disparity;
			                bestMatchIndex = j;
			            }
				    }
				    if(bestMatchIndex!= -1)
				    {
				    	//calculate depth
				    	float bf = 33.4058028773226;
				    	int xLeft = detsLeft[0][i][0].item().toFloat();
				    	int xRight = detsRight[0][bestMatchIndex][0].item().toFloat();
				    	float depth = CalculateDepth(xLeft, xRight, bf);
				    	cout << "depth=" <<depth<< endl;
				    	objectMask(leftBox).setTo(cv::Scalar(1));
				    	AddNewObject(imleft_left,imleft_top,l_left,h_left, classID_left, objectMask, depth);
				    }
		        }
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
