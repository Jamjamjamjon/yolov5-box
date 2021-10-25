#include <sys/time.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "rknn_api.h"
#include <string>
#include <chrono>


/* ==================================
		config [need to modify]
   ==================================*/
#define MAX_OBJ_NUM_PER_FRAME 2000				
#define OBJ_CLASS_NUM     1		//************ classes [need to modify]*************************
#define PROP_BOX_SIZE     (5 + OBJ_CLASS_NUM)

// hps
#define CONF_THRESH 0.3
#define NMS_THRESH 0.3
#define VIDEO_SCALE 0.9										// scale video for displaying and saving 

// input path
// const std::string INPUT = std::string("../data/people.mp4");
// const char* RKNN_MODEL = "../data/weights/yolov5m_416.rknn";		// model path

// classes
/*
static std::string LABELS[OBJ_CLASS_NUM] = {		
			"person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop	","mouse	","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush "
}; 
*/
// static std::string LABELS[OBJ_CLASS_NUM] = {"fall", "normal"};	//for falldown
static std::string LABELS[OBJ_CLASS_NUM] = {"head"};	//for head

// anchors
const int anchor0[6] = {10, 13, 16, 30, 33, 23};
const int anchor1[6] = {30, 61, 62, 45, 59, 119};
const int anchor2[6] = {116, 90, 156, 198, 373, 326};


/* =======================
	structures
   =======================*/
// box rect
typedef struct {
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT;


// bbox: cls, name, box, prop
typedef struct {
	std::string name;
	int cls;
    BOX_RECT box;
    float prop;
} BBOX;


// bboxes : one frame result:
typedef struct {
    int id;
    int count;
    BBOX results[MAX_OBJ_NUM_PER_FRAME];
} BBOXES;


// chrono
struct Timer {
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
	std::chrono::duration<float> duration;

	Timer() {
		start = std::chrono::high_resolution_clock::now();
	}

	~Timer() {
		end = std::chrono::high_resolution_clock::now();
		duration = end - start;
		float ms = duration.count() * 1000.f; 
		std::cout << "==>time took: " << ms << "ms\n";
	}
};


/* =======================
	functions
   =======================*/

// time
double __get_us(struct timeval t);

// load rknn model
unsigned char* load_rknn_model(const char* filename, int* model_size);

/* tensor info */
void RKNNTensorInfo(rknn_tensor_attr* attr);

// post_process
int post_process(uint8_t* input0, uint8_t* input1, uint8_t* input2, int& model_height, int& model_width,
                 float conf_threshold, float nms_threshold, float& scale_w, float& scale_h,
                 std::vector<uint32_t>& qnt_zps, std::vector<float>& qnt_scales,
                 BBOXES& group, int* pad);

// letterbox
cv::Mat letterbox_ori(cv::Mat image, int* pad, int& size_w, int& size_h);
cv::Mat letterbox(cv::Mat image, int* pad, float& scale_w, float& scale_h, int& img_height, int& img_width, int& model_height, int& model_width);

// draw
void draw(cv::Mat& img, BBOXES& bboxes);


