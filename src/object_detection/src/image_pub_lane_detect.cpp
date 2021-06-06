
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sstream> // for converting the command line parameter to integer
#include <torch/script.h>
#include <torch/torch.h>//#define _GLIBCXX_USE_CXX11_ABI 0
#include <iostream>
#include <memory>
#include <algorithm>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <chrono>
#include <sensor_msgs/image_encodings.h>
#include <string>
#include <cmath>
#include <fstream>
// nms library for laneatt
#include "nms/src/nms.cpp"

//importing msg files
#include "geometry_msgs/Point.h"
#include "object_detection/lane_detect_try.h"
#include "object_detection/lane_detect.h"

using namespace torch::indexing;
int num_of_files = 0;
int iterate = 0;
static const std::string OPENCV_WINDOW = "Image window";

class Detector {
    public:
    /***
     * @brief constructor
     * @param model_path - path of the TorchScript weight file
     * @param device_type - inference with CPU/GPU
     */


    //constructor to load the model from torchscript
    Detector(const std::string& model_path, const torch::DeviceType& device_type);

    /***
     * @brief Non Max Supression
     * @param Reg_Proposals - Region Proposals
     * @param conf_threshold - confidence threshold
     * @param attention_matrix - got from attention module of the model
     *
     * @return Proposals_List - the final output for the forward function of the model is proposals list
     */

    std::vector<std::vector<torch::Tensor> > run_the_code(const cv::Mat& img, float conf_threshold);
    //void run_the_code(const cv::Mat& img, float conf_threshold);
    static std::vector<torch::Tensor> non_max_supression(const torch::Tensor& reg_proposals, int nms_thresh, int nms_topk,
					 float conf_threshold);

    /***
     * @brief Decode - It takes the proposals list and then returns the output in the form os lanes and (x,y) coordinates
     * @param Propsals_List - Contains the combination of region proposals and attention matrix
     * @param as_lanes
     * @return Proposals_List - the final output for the forward function of the model is proposals list
     */

    static std::vector<std::vector<torch::Tensor> > decode(const std::vector<torch::Tensor>& proposals_list, bool as_lanes);

    static std::vector<torch::Tensor> proposals_to_pred(const torch::Tensor& proposals);

    torch::jit::script::Module module_;
    torch::Device device_;
    bool half_;

};

//Constructor to load the torchscript model
Detector::Detector(const std::string& model_path, const torch::DeviceType& device_type) : device_(device_type) {
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module_ = torch::jit::load(model_path);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model!\n";
        std::exit(EXIT_FAILURE);
    }

    half_ = (device_ != torch::kCPU);
    module_.to(device_);

    //if (half_) {
      // module_.to(torch::kHalf);
    //}

    module_.eval();
}


// void softmax(double* input, size_t size) 
// {	assert(0 <= size <= sizeof(input) / sizeof(double));	int i;
//   double m, sum, constant;	m = -INFINITY;
//   for (i = 0; i < size; ++i) {
//     if (m < input[i]) {
//       m = input[i];
//     }
//   }	sum = 0.0;
//   for (i = 0; i < size; ++i) {
//     sum += exp(input[i] - m);
//   }	constant = m + log(sum);
//   for (i = 0; i < size; ++i) {
//     input[i] = exp(input[i] - constant);
//   }}
std::vector<std::vector<torch::Tensor> >Detector::run_the_code(const cv::Mat& img, float conf_threshold=0.5) {
 //void Detector::run_the_code(const cv::Mat& img, float conf_threshold=0.5) {
  int count = 180;
  std::ofstream outFile[count];
  std::string name = "/mnt/dheeraj/Lane_Detection/output_agx_";
  std::cout << "----------New Frame----------" << std::endl;
  cv::Mat img_input = img.clone();
  auto comp_start = std::chrono::high_resolution_clock::now();
  auto start = std::chrono::high_resolution_clock::now();
  cv::resize(img_input, img_input, cv::Size(640.0, 360.0));
  //std::cout<<"image is : \n"<< img_input;
  img_input.convertTo(img_input, CV_32FC3, 1.0f / 255.0f);
  //std::cout<<"normalized image is : "<<img_input;
  auto tensor_img = torch::from_blob(img_input.data, {1, img_input.rows, img_input.cols, img_input.channels()}).to(device_);

  tensor_img = tensor_img.permute({0, 3, 1, 2}).contiguous();


  //if (half_) {
    //   tensor_img = tensor_img.to(torch::kHalf);
    //}
  std::vector<torch::jit::IValue> inputs;
  inputs.emplace_back(tensor_img);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
  std::cout << "\n\n preprocessing takes : " << duration.count() << " ms" << std::endl;
  start = std::chrono::high_resolution_clock::now();

  //std::vector<torch::jit::IValue> output = module_.forward(inputs);
   auto output = module_.forward(inputs);
   end = std::chrono::high_resolution_clock::now();

   duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
   std::cout << "\n\n model takes : " << duration.count() << " ms" << std::endl;

  //for (const auto& p : module_.parameters()) {
    //std::cout << p << std::endl;
 // }
  //for (const auto& pair : module_.named_parameters())
  //{
  //std::cout << pair.name << std::endl;
  //}
  //std::cout<<"output is : \n"<<output.toTensor()[0][0];
  //auto end = std::chrono::high_resolution_clock::now();
  start = std::chrono::high_resolution_clock::now();
  auto nms_output = non_max_supression(output.toTensor(), 50, 10, 0.5);

  
  auto prediction = decode(nms_output, true);
  //torch::Tensor ten = torch::rand({12, 12}, torch::TensorOptions(torch::kCPU).dtype(at::kFloat)); 
  //std::cout<<"shape of ten is : "<<ten.sizes()<<std::endl;
  //std::vector<float> v(ten.data_ptr<float>(), ten.data_ptr<float>() + ten.numel()); for (auto a : v) std::cout << a << std::endl;
  //std::cout<<"shape of v is :"<<v.size()<<std::endl; 
  end = std::chrono::high_resolution_clock::now();

  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "post processing takes : " << duration.count() << " ms" << std::endl;
  auto comp_end = std::chrono::high_resolution_clock::now();

  duration = std::chrono::duration_cast<std::chrono::milliseconds>(comp_end - comp_start);
    
  std::cout << "inference takes : " << duration.count() << " ms"<<std::endl;

  //std::cout<< "\n\n FPS is given as :"<<1/(duration/1000)<<std::endl;
  //std::cout<<"prediction shape is : "<<prediction.size()<<std::endl;
  std::cout<<"prediction is :"<<prediction<<std::endl;
  std::cout<<"prediction type is : "<<typeid(prediction).name()<<std::endl;
  //std::cout<<"Prediction shape is :"<<prediction.size()<<std::endl;
  std::cout<<"prediction[0] shape is : "<<prediction[0].size()<<std::endl;
  //std::cout<<"prediction[0][0] shape is : "<<prediction[0][0].sizes()[0]<<std::endl;
 // std::cout<<"x and y coordinates are for 1st lane : "<<prediction[0][0]<<std::endl;
  //std::cout<<"1st x and y coordinate pair for 1st lane is :"<<prediction[0][0][0]<<std::endl;
  // not writing predictions.extend wala part because it is not used anywhere while inferencing
  //need to write annotations wala part somehow
  //outFile<<prediction<<std::endl;
  //outFile[iterate].open(name + std::to_string(num_of_files) + ".txt");
  //outFile[iterate]<<prediction;
  //std::cout<<"filename is given as : "<<std::to_string(num_of_files)<<std::endl;
  //num_of_files = num_of_files+30;
  //iterate = iterate+1;
  //std::cout<<"Shape of prediction is : "<<prediction[0].size()<<std::endl;
  return prediction;
}

std::vector<torch::Tensor> Detector::non_max_supression(const torch::Tensor& reg_proposals, int nms_thresh=50, int nms_topk=1000,
                                    float conf_threshold = 0.5)
{
    std::vector<torch::Tensor> proposals_list, keep_num_to_keep, anchor_inds_copy, proposals_dummy = {}, proposals_dummy_dummy, scores_dummy;
    //torch::Tensor above_threshold;
    std::vector<bool> above_threshold;
    //torch::Tensor NoGradGuard no_grad;
    //std::vector<float>  anchor_inds, anchor_inds_copy, anchor_inds_copy_copy;
    torch::Tensor temp, scores, keep,  _, anchor_inds, anchor_inds_copy_copy;//, scores_copy, anchor_inds_copy;
    std::cout<<"Shape of reg proposals is : "<<reg_proposals.sizes()<<std::endl;
    double array[] = {-1};
    torch::Tensor num_to_keep; //from nms.cpp
    for (int j = 0; j < reg_proposals.sizes()[0]; j++)
    {   torch::Tensor  proposals_copy_copy, proposals_copy, scores_copy, empty_tensor = torch::from_blob(array, {1});
	std::vector<torch::Tensor> empty_vec = {};
	//empty_tensor = torch::stack(empty_vec);
	//std::cout<<"empty tensor is given as : "<<empty_tensor;
        int count=0;
        //std::vector<torch::Tensor> proposals_copy;
        //std::cout<<"Shape of proposals is : "<<reg_proposals[j].sizes();
        anchor_inds = torch::arange(reg_proposals.sizes()[1]);
        temp = torch::nn::functional::softmax(reg_proposals[j].index({Slice(), Slice(None, 2)}),torch::nn::functional::SoftmaxFuncOptions(1));
	scores = temp.index({Slice(), 1});
        //std::cout<<"scores type is : "<<typeid(scores).name()<<"\n\n";
        //std::cout<<"Shape of scores is "<<scores.sizes()<<"\n\n";
        // writing this part just for testing

        //for (int i = 0; i < scores.sizes()[0]; i++)
        //{
           //if(scores[i].item<float>() > 0.5)
	//	std::cout<<"scores are : "<<scores[i];
        //}
        if(conf_threshold != -1.0)
        {  //std::cout<<"Reaching here\n\n";
          for(int i = 0; i < scores.sizes()[0]; i++)
          {
            if (scores[i].item<float>() > conf_threshold)
            {
              above_threshold.push_back(true);
		 //above_threshold = torch::add(true);
	    }
            else
	    {
              above_threshold.push_back(false);
		// above_threshold = torch::add(false);
	    }
          }
       }
       //std::cout<<"above threshold : "<<above_threshold<<"\n\n";
       //std::cout<<"\n\nscores shape is given as : "<<scores.sizes()
       //std::cout<<"Shape of above threshold is : "<<above_threshold.size()<<"\n\n";
       for(int i =0; i < above_threshold.size(); i++)
        {  //std::cout<<"Reaching for loop for above threshold"<<std::endl;
          if(above_threshold[i] == true)
          {  //std::cout<<"reaching inside above threshold for loop  if condition\n\n"<<std::endl;
            proposals_dummy.push_back(reg_proposals[j][i]);
            scores_dummy.push_back(scores[i]);
            anchor_inds_copy.push_back(anchor_inds[i]);
          }
	 else
	  {
		count = count +1;
	  }
       }

       //std::cout<<"\n Proposals dummy here is  : "<<proposals_dummy;
       //std::cout<<"The value of count is : "<<count<<std::endl;
       if (count== above_threshold.size())
	{
		//std::cout<<"Reaching inside this is condition"<<std::endl;
		//proposals_dummy.push_back(empty_tensor);
		proposals_list.push_back(empty_tensor);
		continue;
		//scores_dummy.push_back(empty_tensor);
		//anchor_inds_copy.push_back(empty_tensor);
	}
       //std::cout<<"proposals_dummy : "<<proposals_dummy<<std::endl;
       proposals_copy = torch::stack(proposals_dummy);
       //std::cout<<"proposals_copy is : "<<proposals_copy;
       scores_copy = torch::stack(scores_dummy);
       //std::cout<<"\n\nScores copy shape is given as :\n"<<scores_copy.sizes()<<std::endl;
       //std::cout<<"\n Scores copy given as : \n"<<scores_copy;
       //std::cout<<"\n\n proposals copy type is given as : "<<typeid(proposals_copy).name();
       std::cout<<"\n\n Shape of proposals copy is : "<<proposals_copy.sizes();
       if(reg_proposals[j].sizes()[0] ==0)
	  { //std::cout<<"\n\n Reaching inside reg proposals if condition";
	   proposals_list.push_back(reg_proposals[j][0]);
	   continue;
	  }
       //std::cout<<"\n\n Is it reaching here? ";

       keep_num_to_keep = nms_forward(proposals_copy, scores_copy, nms_thresh, nms_topk);
       keep = keep_num_to_keep[0];
       //std::cout<<"\n\nShape of keep at first : "<<keep.sizes();
       //std::cout<<"\n\n Val of keep before slicing : "<<keep;
       num_to_keep = keep_num_to_keep[1];
       //std::cout<<" \n\n here? ";
       keep = keep.index({Slice(None, num_to_keep.item<int>())});
       //std::cout<<"\n\nshape of keep after slicing : "<<keep.sizes();
       //std::cout<<"\n\n Val of keep after slicing : "<<keep;
       for(int i=0; i < keep.sizes()[0]; i++)
        {
          proposals_dummy_dummy.push_back(proposals_copy[keep[i].item<int>()]);
          //anchor_inds_copy_copy.push_back(anchors[keep[i]]);
        }
       //std::cout<<"\n\nEnd time checking for nms reaching here "; 
       proposals_copy_copy = torch::stack(proposals_dummy_dummy);
       //std::cout<<"\n\nproposals  at the last in nms is : "<<proposals_copy_copy;
    proposals_list.push_back(proposals_copy_copy);
    }
return proposals_list;
}

std::vector<std::vector<torch::Tensor> > Detector::decode(const std::vector<torch::Tensor>& proposals_list, bool as_lanes = true)
{
  //std::cout<<"\n\n\n#############Reaching inside decode function##################";
  //std::cout<<"\n\n Shape of proposals list : "<<proposals_list.size()<<std::endl;
  std::vector<std::vector<torch::Tensor> > decoded;
  std::vector<torch::Tensor> pred, empty_pred= {};
  //std::cout<<"type of  proposals in proposal list is : "<<typeid(proposals_list[0]).name()<<std::endl;
  //std::cout<<"The val of proposals in proposal list is : "<<proposals_list;
  //std::cout<<"length of proposals in proposals list is "<<proposals_list[0].sizes()<<std::endl;
 

 for(int i =0  ; i < proposals_list.size() ; i++ )
  { //std::cout<<"\n\nReaching inside for loop ";
    std::vector<torch::Tensor> temp_pred;
    if (proposals_list[0].sizes()[0] ==1 && proposals_list[0].item<int>()==0)
    {
	decoded.push_back(empty_pred);
        continue;
    }
 
    proposals_list[i].index({Slice(),Slice(None, 2)}) = torch::nn::functional::softmax(proposals_list[i].index({Slice(), Slice(None, 2)}), torch::nn::functional::SoftmaxFuncOptions(1));
    //std::cout<<"\n\nJust here";
    proposals_list[i].index({Slice(), Slice(4)}) = round(proposals_list[i].index({Slice(), Slice(4)}));
    //std::cout<<"\n\nhere again";
    std::vector<torch::Tensor> empty_tensor = {};
    if (proposals_list[i].sizes()[0] == 0)
    { //std::cout<<"\n\nhere in if condition in decode ";
      decoded.push_back(empty_tensor); //append null
      continue;
    }
    std::cout<<"\n\nReached outside if condition";
    if (as_lanes == true)
    { //std::cout<<"\n\nHere in if condition of as lanes is true or not";
      //std::cout<<"\n\nShape of proposals_list here is :"<<proposals_list[i].sizes(); 
      pred = proposals_to_pred(proposals_list[i]);
    }
    else
    {
      temp_pred.push_back(proposals_list[i]);
      pred = temp_pred;
    }
    //std::cout<<"Reached else condition as well";
    decoded.push_back(pred);

  }
  return decoded;
}

std::vector<torch::Tensor> Detector::proposals_to_pred(const torch::Tensor& proposals)
{   
	//std::cout<<"\n\n##########Reached proposals_to pred function as well############# ";
	std::vector<torch::Tensor> lanes;//, lane_xs_dummy, lane_ys_dummy, lane_xs_dummy_dummy, lane_ys_dummy_dummy;
	int n_offsets = 72;
	torch::Tensor anchor_ys = torch::linspace(1, 0, n_offsets);
	//std::cout<<"anchor_ys is : "<<anchor_ys;
	torch::Tensor lane_xs, lane_ys, points;//, lane_xs_copy, lane_ys_copy, points, lane_xs_copy_copy, lane_ys_copy_copy;
	long int start, end, length, img_w, n_strips;
	img_w = 640;
	n_strips = n_offsets -1;
	//
	//std::string lane_start, lane_end, lane;
	//lane_start =  "[Lane]\n";
	//lane_end = "\n[/Lane]";
	bool temp_val ;
        //std::cout<<"\n\n shape of proposals is : "<<proposals.sizes()[0];
	for(int i= 0; i < proposals.sizes()[0]; i++)
	{
		std::vector<bool>  boolean, mask;
		std::vector<torch::Tensor> lane_ys_dummy, lane_xs_dummy, lane_ys_dummy_dummy, lane_xs_dummy_dummy, final_lane;
		torch::Tensor lane_xs_copy,lane_ys_copy,lane_xs_copy_copy, lane_ys_copy_copy;
		lane_xs = proposals[i].index({Slice(5, None)}) /  img_w;
		start = round((proposals[i][2]*n_strips).item<float>());
		//std::cout<<"\n\nstart is : "<<start; 
		length = round(proposals[i][4].item<int>());
		//std::cout<<"\n\nlength is : "<<length;
		end = start + length -1;
		end = std::min(end, anchor_ys.sizes()[0] - 1);
		//std::cout<<"\n\nend is : "<<end;
		//mask
		for(int j = 0; j< start; j++)
		{
			if(lane_xs[j].item<float>() >= 0.0 && lane_xs[j].item<float>() <= 1.0)
		   	{
				temp_val = 1;
				//std::cout<<"Im here   "<<j;
				//break;
			}
		   	else
			{
				temp_val = 0;
			}
		        //std::cout<<"temp val  is : "<<temp_val;
		      	boolean.push_back(1-temp_val);
		}
		      //std::cout<<"\n\nboolean is given as "<<boolean;


		for(int j = 0; j < boolean.size(); j++)
		{
			mask.push_back(boolean[i]);
		}
		//std::cout<<"mask is "<<mask;
		lane_xs.index({Slice(end+1, None)}) = -2;
		//std::cout<<"lane_xs is : "<<lane_xs;
		//std::cout<<"\n\nreaching after lane_xs : "<<lane_xs;
		for (int j = 0; j < mask.size(); j++)
		{
			if (mask[j]==1)
			{
				//lane_xs.index({Slice(None, start)})[j] = -2;
				lane_xs[j] = -2;   //I'm doing this step because the length of start and mask are same.
			}
		}
		//std::cout<<"lane_xs is : "<<lane_xs;
		for( int j = 0; j < anchor_ys.sizes()[0]; j++)
		{
			if (lane_xs[j].item<float>() >=0.0)
			{
				//lane_ys_copy = torch::stack(anchor_ys[j]);
				lane_ys_dummy.push_back(anchor_ys[j]);
				lane_xs_dummy.push_back(lane_xs[j]);
				//lane_xs_copy = torch::stack(lane_xs[j]);
			}
		}
		//std::cout<<"Reaching here? ";
		// for flipping the values
		lane_xs_copy = torch::stack(lane_xs_dummy);
		lane_ys_copy = torch::stack(lane_ys_dummy);
		//std::cout<<"\n\nlane_xs_copy : "<<lane_xs_copy;
		//std::cout<<"\n\n lane_ys_copy : "<<lane_ys_copy;
		//std::cout<<"size of lane_xs_copy is : "<<lane_xs_copy.sizes()[0];
		for(int m = lane_xs_copy.sizes()[0]-1; m >=0 ; m--)
		{
			//std::cout<<"\n\nreaching loop here ";
			//lane_xs_copy_copy = torch::stack(lane_xs_copy[j]);
			//lane_ys_copy_copy = torch::stack(lane_ys_copy[j]);
			lane_xs_dummy_dummy.push_back(lane_xs_copy[m]);
			lane_ys_dummy_dummy.push_back(lane_ys_copy[m]);
		}

		lane_xs_copy_copy = torch::stack(lane_xs_dummy_dummy);
		lane_ys_copy_copy = torch::stack(lane_ys_dummy_dummy);
		//std::cout<<"\n\nlane_xs_copy_copy : "<<lane_xs_copy_copy;
		//std::cout<<"\n\nlane_ys_copy_copy : "<<lane_ys_copy_copy;
		if (lane_xs_copy_copy.sizes()[0] <=1)
		{
			//std::cout<<"inside continue function ";
			continue;
		}
		//Here we need to see what squeeze is and what is dim = 1 in python code
                //std::cout<<"\n\nshape of lane_xs_copy_copy : "<<lane_xs_copy_copy.sizes();
                //std::cout<<"\n\nshape of lane_ys_copy_copy : "<<lane_ys_copy_copy.sizes();
                //std::cout<<"\n Shape of lane_xs when reshaping is done is : "<<lane_xs_copy_copy.view({-1, 1}).sizes();
                lane_xs_copy_copy =lane_xs_copy_copy.view({-1, 1}).to(torch::kCUDA);
                //lane_xs_copy_copy = torch::transpose(lane_xs_copy_copy, 1, 0).to(torch::kCUDA);
                lane_ys_copy_copy =lane_ys_copy_copy.view({-1, 1}).to(torch::kCUDA);
                //lane_ys_copy_copy = torch::transpose(lane_ys_copy_copy, 1, 0).to(torch::kCUDA);
                //std::cout<<"\n Shape of lane_xs when transpose is done is :"<<lane_ys_copy_copy.sizes();
                final_lane.push_back(lane_xs_copy_copy);
                final_lane.push_back(lane_ys_copy_copy);
                //std::cout<<"\n Shape of lane_xs when transpose is done is :"<<lane_ys_copy_copy.sizes();
	 	//points = torch::stack((lane_xs_copy_copy.view({-1, 1}), lane_ys_copy_copy.view({-1, 1})));//.squeeze(2);
		points = torch::stack(final_lane).squeeze(2);
                points = torch::transpose(points, 1, 0);
                // lane = str(points.cpu().numpy())
		lanes.push_back(points);
		//std::cout<<"\n\n points from proposals to pred : "<<points;
                //std::cout<<"\n\n shape of lanes is : "<<lanes.size();
                //std::cout<<"\n\n the value of iterator : "<<i<<std::endl;
	}
        //std::cout<<"hello"<<std::endl;
	//std::cout<<"\n\nFor loop inside proposals_to_pred ended";
	return lanes;
}


class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  ros::Publisher result_pub_;
  torch::DeviceType device_type = torch::kCUDA;
  std::string weights = "/home/nvidia/alive/src/lane_detection/src/traced_laneatt_model_r18.pt";
  Detector detector = Detector(weights, device_type);
	public:
  ImageConverter()
    : it_(nh_)
  {
    // Subscribe to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/kitti/camera_color_left/image_raw", 1,
      &ImageConverter::imageCb, this);
    result_pub_ = nh_.advertise<object_detection::lane_detect_try>("/camera/output_lane", 1000);

    //cv::namedWindow(OPENCV_WINDOW);
    //warmup();
  }

  ~ImageConverter()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    // Perform object detection on frame
    auto result = detector.run_the_code(cv_ptr->image, 0.5);
   auto start = std::chrono::high_resolution_clock::now();
    //std::cout<<"\n\nThe type of results is : "<<typeid(result).name()<<std::endl;
    object_detection::lane_detect_try msg_rec;
    //object_detection::lane_detect final_msg;
    //msg_rec.num_of_lanes = 0;//result[0].size();
    
   //Point my_array[50];
   //std::vector<Point> my_vector_dummy, my_vector_dummy2;
   //std::vector<std::vector<Point> > array_f;
    //Point point;
    //for (int i =0; i < result[0][0].sizes()[0];i++)
    //{
        //std::cout<<"i is " <<i<<std::endl;
      //  point.x = result[0][0][i][0].item<float>();
       // point.y = result[0][0][i][1].item<float>();
        //my_array[i] = point;
        //my_vector_dummy.push_back(point);
    //}
     
    //auto start = std::chrono::high_resolution_clock::now();
    msg_rec.lane.clear();
    int count=0;
    for (int i=0;i < result[0].size();i++)
    {   geometry_msgs::Point point_dummy;
        if(result[0].size() ==0)
	{
	   point_dummy.x = -1;
	   point_dummy.y = -1;
   	   point_dummy.z = -1;
	   msg_rec.lane.push_back(point_dummy);
           msg_rec.num_of_points.push_back(-1);
	   continue;
	}   
	for(int j = 0; j<result[0][i].sizes()[0];j++)
	{
	    point_dummy.x = result[0][i][j][0].item<float>();
	    point_dummy.y = result[0][i][j][1].item<float>();
            point_dummy.z = 0;
            msg_rec.lane.push_back(point_dummy);
	    //count++;
	    //my_vector_dummy2.push_back(point);
	}
	msg_rec.num_of_points.push_back(result[0][i].sizes()[0]);

	//array_f.push_back(my_vector_dummy2);
    }
    
    //std::cout<<"count is : "<<count<<std::endl;

    //std::cout<<"sizeof(my_array) :"<<sizeof(my_array)<<std::endl;
    //std::cout<<"sizeof(Point) :"<<sizeof(Point)<<std::endl;
    //std::cout<<"my_array +sizeof(my_array)/sizeof(Point) : "<<my_array + sizeof(my_array)/sizeof(Point)<<std::endl;
    //std::vector<Point> my_vector (my_array, my_array + sizeof(my_array)/sizeof(Point));
    //std::cout<<"my_vector.x : "<<my_vector.x<<std::endl;
    //std::cout<<"my_array_dummy.x : "<<my_array_dummy.x<<std::endl;
     //msg_rec.lane.clear();
    msg_rec.num_of_lanes = result[0].size();
   auto end = std::chrono::high_resolution_clock::now();

   auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "rostopic processing takes : " << duration.count() << " ms"<<std::endl;
    //std::vector<Point> my_vector (my_array, my_array + sizeof(my_array)/sizeof(Point));
    int  i = 0;
    //std::vector<Point> hello;
    //for (std::vector<Point>::iterator it = my_vector.begin(); it !=my_vector.end(); ++it)
    //{
        //geometry_msgs::Point point;
        //point.x = (*it).x;
        //point.y = (*it).y;
	//point.z = 0;
        //hello.push_back(point)
	//msg_rec.lane.push_back(point);
        //final_msg.lanes.push_back(point);
       // i++;
        //std::cout<<"inside my_vector loop : "<<std::endl;
        //std::cout<<"(*it).x : "<<(*it).x<<std::endl;
    //}
    //for (std::vector<Point>::iterator it=my_vector_dummy.begin(); it !=my_vector_dummy.end(); ++it)
    //{  //std::cout<<"inside my_array_dummy"<<std::endl;
        //geometry_msgs::Point point;
        //point.x = (*it).x;
        //point.y = (*it).y;
        //point.z = 0;
        //hello.push_back(point)
        //msg_rec.lane.push_back(point);
        //final_msg.lanes.push_back(point);
        //i++;
       //std::cout<<"(*it).x : "<<(*it).x<<std::endl;

   //}
    //final_msg.lanes.push_back(hello);

    ROS_INFO("%d", msg_rec.num_of_lanes);
    result_pub_.publish(msg_rec);
 
    //Demo(cv_ptr->image, result, class_names);
    //sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image_od_CV).toImageMsg();
    // Update GUI Window
    //cv::imshow(OPENCV_WINDOW, cv_ptr->image);
    //cv::waitKey(27);

    // Output modified video stream
    //image_pub.publish(msg)
    // image_pub_.publish(cv_ptr->toImageMsg());
  }

void warmup(){
    auto img = cv::imread("/home/nvidia/zidane.jpg");
    std::vector<std::string> fn;
    char filename[50];
    //std::glob("/mnt/dheeraj/Lane_Detection/repositories/LaneATT/datasets/nolabel_dataset/*.jpg")
    //for (auto f:fn)
    //{
    //	auto img = cv::imread(f);
    //	std::cout<<img;
    //}
    for(int i=0;i<=179*30;i = i+30)
    {
        char filename[200];
	sprintf(filename, "/mnt/dheeraj/Lane_Detection/repositories/LaneATT/datasets/nolabel_dataset/%05d.jpg", i);
        auto img_2 = cv::imread(filename);
	std::cout<<filename<<std::endl;
        //break;
	std::cout << "Run once on empty image" << std::endl;
	auto temp_img = cv::Mat::zeros(img.rows, img.cols, CV_32FC3);
	detector.run_the_code(temp_img, 1.0f);
	//for (int i = 0; i < 10; i++) {
//	auto result = detector.run_the_code(img_2,  0.5);
	//}
     }
   }
};



int main(int argc, char **argv)
{
  // Check if video source has been passed as a parameter
  // if(argv[1] == NULL) return 1;

  ros::init(argc, argv, "image_pub_lane_detect");
  ImageConverter ic;
  ros::spin();
  return 0;
}

// need to figure out warmup waali cheez
// need to figure out the predictions k baad kya aur demo ka kaise krna hai
// need to verify softmax slicing in nms function
// check how to append null and verify if tukka works


//defining above_threshold as a vector because cant append a boolean value inside a tensor like this
// How to figure out how to write this in c++ : lane_xs[:start][mask] = -2

