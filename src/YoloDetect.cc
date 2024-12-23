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
		mModule = torch::jit::load("yolo11n-seg.torchscript", torch::kCUDA);
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
		//cout<<"newObjectArea="<< newObject.area<<endl;
		newObject.classID = classID;
		newObject.mapPoints = vector<MapPoint*>(1,static_cast<MapPoint*>(NULL));
		newObject.objectMask = objectMask;
		newObject.depthMinMax = depthMinMax;
		mObjects.push_back(newObject);
	}
	//With the segmentation map 
	void YoloDetect::AddNewObject(cv::Rect2i objectArea,std::string classID, cv::Mat objectSegMap, bool isDynamic)
	{
		std::lock_guard<std::mutex> lock(mMutex);
		Object newObject;
		newObject.area =objectArea;
		newObject.classID = classID;
		newObject.mapPoints = vector<MapPoint*>(1,static_cast<MapPoint*>(NULL));
		newObject.objectMask = objectSegMap;
		if(isDynamic)
			mDynamicObjects.push_back(newObject);
		else
			mObjects.push_back(newObject);
	}

	void YoloDetect::ClearObjects()
	{
		mObjects.clear();
		mDynamicObjects.clear();
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
        if(mImageLeft.empty())
         	return;
	    // Preparing input tensor left
	    cv::resize(mImageLeft, imgLeft, cv::Size(640, 640));
	    cv::cvtColor(imgLeft, imgLeft, cv::COLOR_BGR2RGB);
	    torch::Tensor imgTensorLeft = torch::from_blob(imgLeft.data, {imgLeft.rows, imgLeft.cols,3},torch::kByte).to(torch::kCUDA);
	    imgTensorLeft = imgTensorLeft.permute({2, 0, 1}); //channels, height, width
	    imgTensorLeft = imgTensorLeft.unsqueeze(0);
	    imgTensorLeft = imgTensorLeft.toType(torch::kFloat);
	    imgTensorLeft = imgTensorLeft.div(255.0);
	    //execute inference
	    std::vector<torch::jit::IValue> inputs;
	    imgTensorLeft.to(torch::kCUDA);
	    inputs.push_back(std::move(imgTensorLeft));
	    torch::jit::IValue output = mModule.forward(inputs);
	    //extract predictions
	    auto preds = output.toTuple()->elements();
	    // cout << "preds.size=" <<preds.size()<< endl;

		// for (size_t i = 0; i < preds.size(); ++i) {
		//     torch::Tensor tensor = preds[i].toTensor();
		//     cout << "Tensor " << i << ": " << tensor.sizes() << std::endl;
		// }
		torch::Tensor detections = preds[0].toTensor();
		detections = detections.transpose(1, 2).contiguous();
		torch::Tensor seg_pred = preds[1].toTensor();
		// cout << "seg_pred.sizes" << seg_pred.sizes()<< endl;

		vector<torch::Tensor> det_vector = non_max_suppression_seg(detections, 0.5, 0.6);
		// cout << "det.size =" << det_vector.size() << endl;
	    //similar github https://github.com/kimwoonggon/Cpp_Libtorch_DLL_YoloV8Segmentation_CSharpProject/blob/7fd1386da091fd4c7382ef258c3ac8077af5bbb8/YoloV8DLLProject/dllmain.cpp#L381
		if(det_vector.size()==0)
			return;
		//get the first tensor in the detection vector (we only have one image)
		// Access the first tensor in the detections vector.
 		torch::Tensor det = det_vector[0];
		int size = det.sizes()[0];
		// Initialize an empty segmentation map with the original image dimensions.
		int org_height = mImageLeft.rows; 
		int org_width = mImageLeft.cols;
		cv::Mat total_seg_map = cv::Mat(org_height, org_width, CV_8UC3, cv::Scalar(0, 0, 0));
		torch::Tensor seg_rois;  // Tensor to hold segmentation regions of interest.
		for (int i = 0; i < size; i++) {
		    // Scale bounding box coordinates from the network size to the original
		    // image size.
		    float left = det[i][0].item().toFloat() * org_width /
		                 INPUT_W;  // Ensure left is within image bounds.
		    left = std::max(0.0f, left);  // Ensure left is within image bounds.
		    float top = det[i][1].item().toFloat() * org_height / INPUT_H;
		    top = std::max(top, 0.0f);  // Ensure top is within image bounds.
		    float right = det[i][2].item().toFloat() * org_width / INPUT_H;
		    right = std::min(
		        right,
		        (float)(org_width - 1));  // Ensure right does not exceed image width.
		    float bottom = det[i][3].item().toFloat() * org_height / INPUT_H;
		    bottom = std::min(
		        bottom, (float)(org_height -1));  // Ensure bottom does not exceed image height.
		    float score =det[i][4].item().toFloat();  // Get the detection confidence score.
		    int classID = det[i][37].item().toFloat(); // I'm saving the classID in the last element. 
		    // Assign detection properties to the objects array.
		    cv::Rect2i objectArea(left, top, right - left, bottom - top);
		  	//cout << "objectArea = " << objectArea<< endl;
		  	//cout << "classID=" << mClassnames[classID]<< endl;
		  	seg_rois = det[i].slice(0, 5, det[i].sizes()[0]-1);  // Extract segmentation ROI.(the latest element is the classID)
		  	seg_rois = seg_rois.view({1, 32}).to(torch::kCUDA);
		  	seg_pred = seg_pred.to(torch::kCUDA);
		  	seg_pred = seg_pred.view({1, 32, -1});
		  	auto final_seg = torch::matmul(seg_rois, seg_pred).view({1, 160, 160});
		  	final_seg = final_seg.sigmoid();  // Apply sigmoid to get mask probabilities.
		  	float _seg_thresh = 0.5f;
		  	final_seg = ((final_seg > _seg_thresh) * 255).clamp(0, 255).to(torch::kCPU).to(torch::kU8);
		  	// Convert probabilities to binary mask.
		  	cv::Mat seg_map(160, 160, CV_8UC1,final_seg.data_ptr());
		  	//back to the original size.
		    cv::Mat object_seg_map;
		    cv::resize(seg_map, object_seg_map, cv::Size(org_width, org_height),
		               cv::INTER_LINEAR);  	
		  	if(mClassnames[classID]=="person"){
		  	// 	cv::namedWindow("Segmentation Map", cv::WINDOW_NORMAL);
		  	// 	cv::imshow("Segmentation Map", object_seg_map);
		  	// }
		  	// cv::waitKey(0);
		  		AddNewObject(objectArea, mClassnames[classID],object_seg_map, true);
		  	}
		  	else {
		  		AddNewObject(objectArea, mClassnames[classID],object_seg_map, false);
		  	}
		}
	}


	std::vector<YoloDetect::Object> YoloDetect::GetObjects()
    {
       	std::lock_guard<std::mutex> lock(mMutex); 
        return mObjects;
    }
	std::vector<YoloDetect::Object> YoloDetect::GetDynamicObjects()
    {
        std::lock_guard<std::mutex> lock(mMutex); 
        return mDynamicObjects;
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
        // cout << "output="<< output<< endl;;
    }

    return output;
}

vector<torch::Tensor> YoloDetect::non_max_suppression_seg(torch::Tensor preds, float score_thresh, float iou_thresh)
{
	std::vector<torch::Tensor> output;
	for (size_t i = 0; i < preds.sizes()[0]; ++i) {
		 torch::Tensor pred = preds.select(0, i).to(torch::kCUDA);
		 torch::Tensor scores = std::get<0>(torch::max(pred.slice(1, 4, 84), 1));
		 auto mask = scores > score_thresh;
		 if (mask.sum().item<int>() > 0) {
		 	torch::Tensor indices = torch::nonzero(mask).select(1, 0).to(torch::kCUDA);
		 	pred = torch::index_select(pred, 0, indices);
		    // Convert bounding box format from center x, center y, width, height (cx,
		    // cy, w, h) to top-left and bottom-right corners (x1, y1, x2, y2).
		 	//pred boxes
		    pred.select(1, 0) =
		        pred.select(1, 0) - pred.select(1, 2) / 2;  // Calculate x1
		    pred.select(1, 1) =
		        pred.select(1, 1) - pred.select(1, 3) / 2;              // Calculate y1
		    pred.select(1, 2) = pred.select(1, 0) + pred.select(1, 2);  // Calculate x2
		    pred.select(1, 3) = pred.select(1, 1) + pred.select(1, 3);  // Calculate y2
		    // Identify the maximum confidence score for each prediction and its
		    // corresponding class.
		    auto max_tuple = torch::max(pred.slice(1, 4, 84), 1);
		    pred.select(1, 4) = std::get<0>(max_tuple);  // Set max confidence score
			torch::Tensor predLoc = std::get<1>(max_tuple).to(pred.device());  // Store class id
		    torch::Tensor dets;
		    // Combine bounding box coordinates with confidence scores and class ids
		    // into a single tensor.
		    dets = torch::cat({pred.slice(1, 0, 5), pred.slice(1, 84, 116), predLoc.unsqueeze(1)}, 1);
		    // Prepare tensors to keep track of indices of detections to retain.
			torch::Tensor keep = torch::empty({dets.sizes()[0]}, torch::kInt64).to(torch::kCUDA);
		    torch::Tensor areas = (dets.select(1, 3) - dets.select(1, 1)) *
                          		  (dets.select(1, 2) - dets.select(1, 0)).to(torch::kCUDA);
		    // Sort detections by confidence score in descending order.
		    auto indexes_tuple = torch::sort(dets.select(1, 4), 0,
		                                     1);  // 0: first order, 1: decending order
		    torch::Tensor v = std::get<0>(indexes_tuple);
		    torch::Tensor indexes = std::get<1>(indexes_tuple);

		    int count = 0;  // Counter for detections to keep.
		    // Loop over detections and apply non-maximum suppression.
		    while (indexes.sizes()[0] > 0) {
		    	// Always keep the detection with the highest current score.
			    keep[count++] = indexes[0].item<int64_t>();
			    // Compute the pairwise overlap between the highest scoring detection and
			      // all others. Preallocate tensors to hold the computed overlaps.
			      torch::Tensor lefts =
			          torch::empty(indexes.sizes()[0] - 1, indexes.options());
			      torch::Tensor tops =
			          torch::empty(indexes.sizes()[0] - 1, indexes.options());
			      torch::Tensor rights =
			          torch::empty(indexes.sizes()[0] - 1, indexes.options());
			      torch::Tensor bottoms =
			          torch::empty(indexes.sizes()[0] - 1, indexes.options());
			      torch::Tensor widths =
			          torch::empty(indexes.sizes()[0] - 1, indexes.options());
			      torch::Tensor heights =
			          torch::empty(indexes.sizes()[0] - 1, indexes.options());

			      // Loop over each detection remaining after the one with the highest
			      // score.
			      for (size_t i = 0; i < indexes.sizes()[0] - 1; ++i) {
			        // Compute the coordinates of the intersection rectangle.
			      	lefts[i] = std::max(dets[indexes[0]][0].item().toFloat(),
			                          dets[indexes[i + 1]][0].item().toFloat());
			        tops[i] = std::max(dets[indexes[0]][1].item().toFloat(),
			                           dets[indexes[i + 1]][1].item().toFloat());
			        rights[i] = std::min(dets[indexes[0]][2].item().toFloat(),
			                             dets[indexes[i + 1]][2].item().toFloat());
			        bottoms[i] = std::min(dets[indexes[0]][3].item().toFloat(),
			                              dets[indexes[i + 1]][3].item().toFloat());
			        widths[i] = std::max(
			            float(0), rights[i].item().toFloat() - lefts[i].item().toFloat());
			        heights[i] = std::max(
			            float(0), bottoms[i].item().toFloat() - tops[i].item().toFloat());
			      }

			      // Compute the intersection over union (IoU) for each pair.
			      torch::Tensor overlaps = widths * heights;
			      torch::Tensor ious =
			          overlaps / (areas.select(0, indexes[0].item().toInt()) +
			                      torch::index_select(
			                          areas, 0, indexes.slice(0, 1, indexes.sizes()[0])) -
			                      overlaps);
			      auto nonzero_indices = torch::nonzero(ious <= iou_thresh);
			      torch::Tensor kk = torch::nonzero(ious <= iou_thresh).select(1, 0) + 1;
			      // Filter out detections with IoU above the threshold, as they overlap too
			      // much with the highest scoring box.
			      indexes = torch::index_select(indexes, 0, torch::nonzero(ious <= iou_thresh).select(1, 0) + 1);
		    }
		    keep = keep.slice(0, 0, count);
		    predLoc = torch::index_select(predLoc, 0, keep).to(torch::kCPU);
		    // cout << "count = " << count << endl;
		    //select the keep detections to add to the output 
		    output.push_back(torch::index_select(dets, 0, keep));
		    // cout << "output.size=" << output[0].sizes()<< endl;
		 } else {
		 	continue;
		 }
	}
	return output;
}

    void YoloDetect::GetImage(cv::Mat &imageLeft, cv::Mat &imageRight)
	{
		std::lock_guard<std::mutex> lock(mMutex);
    	mImageLeft = imageLeft;
    	mImageRight = imageRight;
	}
    void YoloDetect::GetImage(cv::Mat &imageLeft)
	{
		std::lock_guard<std::mutex> lock(mMutex);
    	mImageLeft = imageLeft;
	}
	void YoloDetect::Run()
	{
		while(true)
		{
			Detect();
			usleep(1000);
		}
	}
} //namespace ORB_SLAM3
