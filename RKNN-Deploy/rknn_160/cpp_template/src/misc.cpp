#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>				
#include <string>
#include <sys/time.h>
#include <vector>
#include <stdint.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "rknn_api.h"
#include "misc.h"
#include <boost/algorithm/clamp.hpp>
#include <array>



/* linux strcut timeval  */
double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }


// new @jamjon
unsigned char* load_rknn_model(const char* filename, int* model_size) {
    FILE* fp;
    unsigned char* data = NULL;
	int ret;

    fp = fopen(filename, "rb");
    if (NULL == fp) {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    ret = fseek(fp, 0, SEEK_SET);
    if (ret != 0) {
        printf("blob seek failure.\n");
        return NULL;
    }
    data = (unsigned char*)malloc(size);
    if (data == NULL) {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, size, fp);
	fclose(fp);
    *model_size = size;
    return data;
}


/* tensor info */
void RKNNTensorInfo(rknn_tensor_attr* attr) {
    printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n",
           	attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0], attr->n_elems, 
		  	attr->size, attr->fmt, attr->type, attr->qnt_type, 
		   	attr->fl, attr->zp, attr->scale);    // fl(int8_t) zp(uint32_t) scale(float) => qnt parameters
}



/* IOU */
static float IOU(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1, float ymax1) {	
    float w = std::max(0.f, std::min(xmax0, xmax1) - std::max(xmin0, xmin1) + 1.0f);
    float h = std::max(0.f, std::min(ymax0, ymax1) - std::max(ymin0, ymin1) + 1.0f);
    float i = w * h;	// intersection
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;	// union
    return u <= 0.f ? 0.f : (i / u);
}


/* NMS */
// outputLocations : valid bboxes
// order : bboxes score order
static int NMS(int validCount, std::vector<float>& outputLocations, std::vector<int>& order, float threshold) {
    for (int i = 0; i < validCount; ++i) {
        if (order[i] == -1) {
            continue;
        }
        int n = order[i];
        for (int j = i + 1; j < validCount; ++j) {
            int m = order[j];
            if (m == -1) {
                continue;
            }
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = IOU(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold) {
                order[j] = -1;
            }
        }
    }
    return 0;
}

// NMS_2
static int NMS_2(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order, int filterId, float threshold) {
    for (int i = 0; i < validCount; ++i) {
        if (order[i] == -1 || classIds[i] != filterId) {
            continue;
        }
        int n = order[i];
        for (int j = i + 1; j < validCount; ++j) {
            int m = order[j];
            if (m == -1 || classIds[i] != filterId) {
                continue;
            }
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = IOU(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold) {
                order[j] = -1;
            }
        }
    }
    return 0;
}

/* quick sort */
// with indexArray
static int quick_sort(std::vector<float> &input, int left, int right, std::vector<int> &indices) {
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right) {
        key_index = indices[left];
        key = input[left];
        while (low < high) {
            while (low < high && input[high] <= key) {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key) {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort(input, left, low - 1, indices);
        quick_sort(input, low + 1, right, indices);
    }
    return low;
}



/* sigmoid */
// <math.h> => c   , <cmath> => c++
static float sigmoid(float x) {
    return 1.0 / (1.0 + expf(-x));
}

// unsigmoid 
static float unsigmoid(float y) {
    return -1.0 * logf((1.0 / y) - 1.0);
}


/* inline to replace define */
// int 
// #include <boost/algorithm/clamp.hpp>   ==>   boost::algorithm::clamp
inline static int clamp(float val, int min, int max) {
    return val > min ? (val < max ? val : max) : min;   // val => [min, val ,max]
}


// int32
inline static int32_t __clip(float val, float min, float max) {
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

// qnt : float32 -> uint8;  scale: qnt affine asymmetric
// typedef unsigned char uint8_t
// typedef unsigned short int uint16_t
// typedef unsigned int uint32_t
static uint8_t qnt_f32_to_affine(float f32, uint32_t zp, float scale) {
    float dst_val = (f32 / scale) + zp;
    uint8_t res = (uint8_t)__clip(dst_val, 0, 255);
    return res;
}

// deqnt :  uint8 -> float32
static float deqnt_affine_to_f32(uint8_t qnt, uint32_t zp, float scale) {
    return ((float)qnt - (float)zp) * scale;
}


//process 1: for shape like [(1,255,80,80), (1,255,40,40), (1,255,20,20)]
static int process_1(uint8_t* input, int* anchor, int grid_h, int grid_w, int height, int width, int stride,
                   std::vector<float>& boxes, std::vector<float>& boxScores, std::vector<int>& classId,
                   float threshold, uint32_t zp, float scale) {

    int validCount = 0;
    int grid_len = grid_h * grid_w;
    float thres = unsigmoid(threshold);
    uint8_t thres_u8 = qnt_f32_to_affine(thres, zp, scale);

    for (int a = 0; a < 3; a++) {
        for (int i = 0; i < grid_h; i++) {
            for (int j = 0; j < grid_w; j++) {
                uint8_t box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
				
                if (box_confidence >= thres_u8) {
                    int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
                    uint8_t *in_ptr = input + offset;
                    float box_x = sigmoid(deqnt_affine_to_f32(*in_ptr, zp, scale)) * 2.0 - 0.5;
                    float box_y = sigmoid(deqnt_affine_to_f32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
                    float box_w = sigmoid(deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
                    float box_h = sigmoid(deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = box_w * box_w * (float)anchor[a * 2];
                    box_h = box_h * box_h * (float)anchor[a * 2 + 1];
                    box_x -= (box_w / 2.0);
                    box_y -= (box_h / 2.0);
                    boxes.emplace_back(box_x);
                    boxes.emplace_back(box_y);
                    boxes.emplace_back(box_w);
                    boxes.emplace_back(box_h);

                    uint8_t maxClassProbs = in_ptr[5 * grid_len];
                    int maxClassId = 0;
                    for (int k = 1; k < OBJ_CLASS_NUM; ++k) {
                        uint8_t prob = in_ptr[(5 + k) * grid_len];
                        if (prob > maxClassProbs) {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }
					boxScores.emplace_back(sigmoid(deqnt_affine_to_f32(maxClassProbs, zp, scale)));
                    classId.emplace_back(maxClassId);
                    validCount++;
                }
            }
        }
    }
    return validCount;
}



// process 2: for shape like [(1,3,80,80,85), ...]
static int process_2(uint8_t* input, int* anchor, int grid_h, int grid_w, int height, int width, int stride,
                   std::vector<float> &boxes, std::vector<float> &boxScores, std::vector<int> &classId,
                   float threshold, uint8_t zp, float scale) {

    int validCount = 0;
    int grid_len = grid_h * grid_w;		// 80 * 80
    float thres = unsigmoid(threshold);		// 0.4 ==>  -0.405465
    uint8_t thres_u8 = qnt_f32_to_affine(thres, zp, scale);


	// output dims=[255 80 80]
	// input shape : [255 80 80]
	// grid_h / grid_w =>  80,40,20 
	// height/width => 640
	// stride = 8,16,32
    for (int a = 0; a < 3; a++)	{	// every grid predict 3 bbox
        for (int i = 0; i < grid_h; i++) {	// for every grid: row 
            for (int j = 0; j < grid_w; j++) {	// for every grid: col{
				// confidence
                //uint8_t box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];

				/*----------- new update ------------- */ 
				int pos = a * grid_w * grid_h * PROP_BOX_SIZE + i * grid_w * PROP_BOX_SIZE + j * PROP_BOX_SIZE;
                uint8_t box_confidence = input[4 + pos];
			
                if (box_confidence >= thres_u8) {
      
					// -----------------------------------------------------------------
					/* for shape [3,80,80,85]  */
					// to do !!!! ----------------------------
					float box_x = sigmoid(deqnt_affine_to_f32(input[0 + pos], zp, scale)) * 2.0 - 0.5;
                    float box_y = sigmoid(deqnt_affine_to_f32(input[1 + pos], zp, scale)) * 2.0 - 0.5;
                    float box_w = sigmoid(deqnt_affine_to_f32(input[2 + pos], zp, scale)) * 2.0;
                    float box_h = sigmoid(deqnt_affine_to_f32(input[3 + pos], zp, scale)) * 2.0;
					

					// ----------------------------------------------------------------

                    box_x = (box_x + j) * (float)stride;				// center x
                    box_y = (box_y + i) * (float)stride;				// center y
                    box_w = box_w * box_w * (float)anchor[a * 2];		// w, width
                    box_h = box_h * box_h * (float)anchor[a * 2 + 1];	// h, height

                    box_x -= (box_w / 2.0);		// left
                    box_y -= (box_h / 2.0);		// top

                    boxes.emplace_back(box_x);
                    boxes.emplace_back(box_y);
                    boxes.emplace_back(box_w);
                    boxes.emplace_back(box_h);

                    //uint8_t maxClassProbs = in_ptr[5 * grid_len];	// class prob    
        			float maxClassProbs = 0;	// modified ----------------------------------

                    int maxClassId = 0;
                    for (int k = 0; k < OBJ_CLASS_NUM; ++k) {
                        // uint8_t prob = in_ptr[(5 + k) * grid_len];


						// to do 
						uint8_t pred_cla = input[k + 5 + pos];					// modified ----------------------------------
            			float prob = box_confidence * pred_cla;					// uint8_t ????

                        if (prob > maxClassProbs) {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }
					boxScores.emplace_back(sigmoid(deqnt_affine_to_f32(maxClassProbs, zp, scale)));
                    classId.emplace_back(maxClassId);		// class id
                    validCount++;
                }
            }
        }
    }
    return validCount;
}


// post-process
int post_process(uint8_t* input0, uint8_t* input1, uint8_t* input2, int& model_height, int& model_width,
                 float conf_threshold, float nms_threshold, float& scale_w, float& scale_h,
                 std::vector<uint32_t>& qnt_zps, std::vector<float>& qnt_scales,
                 BBOXES& group, int* pad) {


    memset(&group, 0, sizeof(BBOXES));

    std::vector<float> filterBoxes;	
    std::vector<float> boxesScore;
    std::vector<int> classId;

    int stride0 = 8;
    int grid_h0 = model_height / stride0;		// 640 / 8 = 80
    int grid_w0 = model_width / stride0;		// 640 / 8 = 80
    int validCount0 = 0;
    validCount0 = process_1(input0, (int*)anchor0, grid_h0, grid_w0, model_height, model_width,
                          stride0, filterBoxes, boxesScore, classId, conf_threshold, qnt_zps[0], qnt_scales[0]);

    int stride1 = 16;
    int grid_h1 = model_height / stride1;
    int grid_w1 = model_width / stride1;
    int validCount1 = 0;
    validCount1 = process_1(input1, (int*)anchor1, grid_h1, grid_w1, model_height, model_width,
                          stride1, filterBoxes, boxesScore, classId, conf_threshold, qnt_zps[1], qnt_scales[1]);

    int stride2 = 32;
    int grid_h2 = model_height / stride2;
    int grid_w2 = model_width / stride2;
    int validCount2 = 0;
    validCount2 = process_1(input2, (int*)anchor2, grid_h2, grid_w2, model_height, model_width,
                          stride2, filterBoxes, boxesScore, classId, conf_threshold, qnt_zps[2], qnt_scales[2]);

    int validCount = validCount0 + validCount1 + validCount2;	// all objects

    // no object detect
    if (validCount <= 0) {
        return 0;
    }
	
	// num obejct -> indexArray [0,1,...,validCount]
    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i) {
		indexArray.emplace_back(i);
    }

	// quick sort by box score, generate indexArray
    quick_sort(boxesScore, 0, validCount - 1, indexArray);
	
	// new NMS parts @jamjon 1018
    std::set<int> class_set(std::begin(classId), std::end(classId));
    for (auto c : class_set) {
        NMS_2(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
    }

    // NMS(validCount, filterBoxes, indexArray, nms_threshold);


    int last_count = 0;
    group.count = 0;


    /* box valid detect target */
    for (int i = 0; i < validCount; ++i) {
		if (indexArray[i] == -1 || i >= MAX_OBJ_NUM_PER_FRAME) {
			// std::cout << "==>It's too much valid bboxes to post-process in this frame!!!\n";
            continue;
        }

        int n = indexArray[i];

		
		
        float x1 = filterBoxes[n * 4 + 0] - pad[1]; 	// PAD[1] => PAD_W
        float y1 = filterBoxes[n * 4 + 1] - pad[0]; 	// PAD[0] => PAD_H
        float x2 = x1 + filterBoxes[n * 4 + 2] + 0.5;
        float y2 = y1 + filterBoxes[n * 4 + 3] + 0.5;
		
		if (scale_w < scale_h) {
			group.results[last_count].box.left = (int)(clamp(x1, 0, model_width) / scale_w);
		    group.results[last_count].box.top = (int)(clamp(y1, 0, model_height) / scale_w);
		    group.results[last_count].box.right = (int)(clamp(x2, 0, model_width) / scale_w);
		    group.results[last_count].box.bottom = (int)(clamp(y2, 0, model_height) / scale_w);
		} else {
			group.results[last_count].box.left = (int)(clamp(x1, 0, model_width) / scale_h);
		    group.results[last_count].box.top = (int)(clamp(y1, 0, model_height) / scale_h);
		    group.results[last_count].box.right = (int)(clamp(x2, 0, model_width) / scale_h);
		    group.results[last_count].box.bottom = (int)(clamp(y2, 0, model_height) / scale_h);

		}
        
		int id = classId[n];
        group.results[last_count].prop = boxesScore[i];
		std::string label = std::string(LABELS[id]);
		group.results[last_count].name = label;
		group.results[last_count].cls = id;

		last_count++;
    }
    group.count = last_count;
	
	// printf("==>num_object: %d , (filtered from %d)\n", last_count, validCount);	// *************
	// printf("==>num_object: %d\n", last_count);	// *************    
	return 0;
}




// letterbox
cv::Mat letterbox_ori(cv::Mat image, int* pad, int& size_w, int& size_h) {

    int ori_h = image.rows;
    int ori_w = image.cols;

	float scale_w = (float)size_w / ori_w; 	// 416/ori
	float scale_h = (float)size_h / ori_h;

	float r = std::min(scale_w, scale_h);
	int new_unpad_w = int(std::ceil(r * ori_w));
	int new_unpad_h = int(std::ceil(r * ori_h));


	pad[1] = std::max(0, size_w - new_unpad_w) / 2;
	pad[0] = std::max(0, size_h - new_unpad_h) / 2;

	cv::resize(image, image, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);
	cv::copyMakeBorder(image, image, pad[0], pad[0], pad[1], pad[1], cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

	// cv::imshow("111", image);
	// cv::waitKey(0);	

    return image;
}



// letterbox
cv::Mat letterbox(cv::Mat image, int* pad, float& scale_w, float& scale_h, int& img_height, int& img_width,  int& model_height, int& model_width) {
    

    float r = std::min(scale_w, scale_h);
    int new_unpad_w = int(std::ceil(r * img_width));
    int new_unpad_h = int(std::ceil(r * img_height));

    pad[0] = std::max(0, model_height - new_unpad_h) / 2;
    pad[1] = std::max(0, model_width - new_unpad_w) / 2;

    cv::resize(image, image, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);
    cv::copyMakeBorder(image, image, pad[0], pad[0], pad[1], pad[1], cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    // cv::imshow("111", image);
    // cv::waitKey(0);  

    return image;
}



// draw function
void draw(cv::Mat& img, 
		 BBOXES& bboxes
		 // cv::Scalar bbox_color, 
		 // int bbox_thickness, 
		 // cv::Scalar label_color, 
		 // int line_thickness, 
		 // float font_scale, 
		 // bool display_res,
		 // bool simplify
		 ) { 
	for (int i = 0; i < bboxes.count; i++) {
			BBOX* det_bbox = &(bboxes.results[i]);


			// printf("\n%s [%d] (%.3f) | [%d, %d, %d, %d]\n", det_bbox->name, det_bbox->cls, det_bbox->prop, det_bbox->box.left, det_bbox->box.top, det_bbox->box.right, det_bbox->box.bottom);


			int x1 = det_bbox->box.left;
			int y1 = det_bbox->box.top;
			int x2 = det_bbox->box.right;
			int y2 = det_bbox->box.bottom;
			std::string name = det_bbox->name;

			char prop_str[4];	
			sprintf(prop_str, "%.2f", (det_bbox->prop) * 100);	
			std::string label = name + std::string(" ") + std::string(prop_str) + std::string("%");		
			
			rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0,123,155,255), 2);
			putText(img, label, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(4,255,255,255), 2, 8);	

	}

			/*

			if (simplify) {
				rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0,255,255,255), 2);
				putText(img, det_bbox->name, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,255,255), 2, 8);

			} else {

				// label = name + prop
				char* prop_str;
				float p = det_bbox->prop;				
				sprintf(prop_str, "%.2f", p*100);
				std::string space = " ";
				std::string percentage = "%";			
				std::string label = det_bbox->name + space + std::string(prop_str) + percentage;
				
				
				if (det_bbox->name == "people") {
					// bbox & text
					rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0,0,255,255), 2);
					putText(img, label, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255,255), 2, 8);		
				} else {
					// bbox & text
					rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 255, 255), 2);
					putText(img, label, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255, 255), 2, 8);

				}

			}*/	
			
		
		

}


