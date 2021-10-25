/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "rknn_api.h"
#include "misc.h"
#include "omp.h"


/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char **argv)
{
	
	Timer t;					// chrono time
    rknn_context ctx;			// rknn ctx
    int ret;					// rknn return value
    int model_channel = 3;		//rknn model input shape(x, x, 3) 
    int model_width = 0;
    int model_height = 0;
	//std::array<int, 2> PAD = {0, 0};
	int PAD[2] = {0, 0}; 		// pad_h, pad_w use for letter_box()


	// [optional] for testing
    if (argc != 3)
    {
        printf("Usage: %s <rknn model> <image/video> \n", argv[0]);
        return -1;
    }

    char* RKNN_MODEL = (char *)argv[1];
    std::string INPUT = argv[2];

    
	// [optional] check if there is a video
    cv::VideoCapture cap(INPUT);
    if (!cap.isOpened()) {
        std::cout << "video can not be opened." << std::endl;
        exit(1);
    }

    // [step_1] Create the neural network
    int model_data_size = 0;
	unsigned char* model_data = load_rknn_model(RKNN_MODEL, &model_data_size);  // @jamjon 1018
	printf("==>Model loaded.\n");

	// [step_2] RKNN init
    ret = rknn_init(&ctx, model_data, model_data_size, RKNN_FLAG_ASYNC_MASK);			// RKNN_FLAG_ASYNC_MASK
    if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
	printf("==>RKNN init succeed.\n");

	// [step_3] SDK version 
    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    // printf("==>SDK version: %s driver version: %s\n", version.api_version, version.drv_version);

	// [step_4] Rknn input num & output num ==>  rknn_input_output_num 
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    // printf("==>model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

	// [step_5] input attr ==> rknn_tensor_attr 
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
        RKNNTensorInfo(&(input_attrs[i]));
    }

	// [step_6] output attr
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        RKNNTensorInfo(&(output_attrs[i]));
    }	
	
	// [step_7] rknn shape format 
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        printf("==>model is NCHW input fmt\n");
        model_width = input_attrs[0].dims[0];
        model_height = input_attrs[0].dims[1];
    } else {
        printf("==>model is NHWC input fmt\n");
        model_width = input_attrs[0].dims[1];
        model_height = input_attrs[0].dims[2];
    }
    // printf("==>model input height=%d, width=%d, channel=%d\n", model_height, model_width, model_channel);
	

	// [step_8] model input data setting
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = model_width * model_height * model_channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = false;  //  = 1 or True ==> it will give the data in inputs[0].buf directly to model's input node, and do nothing pre-process at all; // used with pre-compile in the process of converting to rknn model.(maybe)


	// [step_9] output sets
	rknn_output outputs[io_num.n_output];			
	memset(outputs, 0, sizeof(outputs));
	for (int i = 0; i < io_num.n_output; i++) {
		outputs[i].want_float = false;				//  use uint8_t type or float type; if set to 1 or True, include dequant process
		// outputs[i].is_prealloc = false; 			// = 1, user decide where to allocate buffer and release Mem by himself; = 0, rknn auto mode. 
	}
	
	// [step_10] qnt params
	std::vector<float> out_scales;		
	std::vector<uint32_t> out_zps;		
	for (int i = 0; i < io_num.n_output; ++i) {
		out_scales.emplace_back(output_attrs[i].scale);
		out_zps.emplace_back(output_attrs[i].zp);
	}
    

	// [step_11] opencv read video
	std::cout << "==>Build with opencv: " << CV_VERSION << std::endl;
	cv:: VideoCapture capture;	// capture instance
	capture.open(INPUT);	// open video

	// [step_12] video_writer to get video info
	cv::VideoWriter video_writer;
	int fourcc = cv::VideoWriter::fourcc('I','4','2','0');
	double fps = capture.get(CV_CAP_PROP_FPS);
	int img_width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	int img_height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	printf("==>Media input: img width = %d, img height = %d\n", img_width, img_height);

	// [optional] save video
	// video_writer.open("../out_video.avi", fourcc, fps, cv::Size(int(img_width * VIDEO_SCALE), int(img_height * VIDEO_SCALE)), true);
	// printf("==>video writer load!");

	// [step_13] calc scales
	float scale_w = (float)model_width / img_width;
	float scale_h = (float)model_height / img_height;

	// [optional] for time testing
	struct timeval start_time, stop_time_get_output, stop_time_infer, stop_time, stop_time_preprocess, stop_time_postprocess;	

	// [step_14] detecting video
	if (capture.isOpened()) {
		cv::Mat frame;		// original frame  
		cv::Mat image;		// used as a temp variable for drawing
		// ==> post process
		BBOXES det_bboxes;			// [id, count, bbox, name] in one frame

		// #pragma omp parallel for 
		for(;;) {  // { Timer t1;
			gettimeofday(&start_time, NULL); // start time
			
			// [step_15] read frame
			//capture >> frame;	// read frame  capture.read(frame)
			capture.read(frame);			
			if (frame.empty()) break;
			
			// [step_16] frame pre-process
			// image = letterbox_ori(frame, PAD, model_width, model_height); 	// letterbox() to resize
			image = letterbox(frame, PAD, scale_w, scale_h, img_height, img_width, model_height, model_width);   // new @jamjon 1018
			// gettimeofday(&stop_time_preprocess, NULL);
			// printf("==>read image + resize consume: %f ms\n", (__get_us(stop_time_preprocess) - __get_us(start_time)) / 1000);

			// [step_17] inputs set
			inputs[0].buf = image.data;						
			rknn_inputs_set(ctx, io_num.n_input, inputs);
			
			// [step_18] inference
			ret = rknn_run(ctx, NULL);		
			// gettimeofday(&stop_time_infer, NULL);
			// printf("==>rknn_run() consume: %f ms\n", (__get_us(stop_time_infer) - __get_us(stop_time_preprocess)) / 1000);
			
			// [step_19] get model outputs: 
			ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL); 	// API = 1.7.x : rknn_outputs_map() can reduce Mem copy!
			// gettimeofday(&stop_time_get_output, NULL);
			// ("==>rknn_outputs_get() consume: %f ms\n", (__get_us(stop_time_get_output) - __get_us(stop_time_infer)) / 1000);

			// [step_20] input type => uint8_t
			post_process((uint8_t *)outputs[0].buf, (uint8_t *)outputs[1].buf, (uint8_t *)outputs[2].buf, 
						  model_height, model_width,
				          CONF_THRESH, NMS_THRESH, 
						  scale_w, scale_h, out_zps, out_scales, 
						  det_bboxes, PAD);
	

			// [step_21] Draw bbox 
			draw(frame, det_bboxes);	
			// gettimeofday(&stop_time_postprocess, NULL);
			// printf("==>post-process + draw consume: %f ms\n", (__get_us(stop_time_postprocess) - __get_us(stop_time_get_output)) / 1000);

			// [step_22] clear current output
			ret = rknn_outputs_release(ctx, io_num.n_output, outputs);	// old one	
			
			// [optional] resize frame to show
			// cv::resize(frame, frame, cv::Size(int(img_width * VIDEO_SCALE), int(img_height * VIDEO_SCALE)), 0, 0, cv::INTER_LINEAR);
			
			// [optional] show every frame after draw
			cv::imshow("opencv_display", frame);

			// [optional] write frame to video
			// video_writer << frame;
			
			// [optional] save last image
			// cv::imwrite("../out.jpg", frame);

			// stop time
			gettimeofday(&stop_time, NULL);		
			double frame_t = (__get_us(stop_time) - __get_us(start_time)) / 1000;
			printf("==>Frame consume: %.2f ms\n", frame_t);
			//	printf("==>FPS: %f \n", 1 / frame_t);

			// sep
			// std::cout << "======================================\n";

			// exit mannually
			if (cv::waitKey(1) == 'q') break;
		
		}
	} else {
		std::cout << "==>No video captured." << std::endl;	
		exit(1);
	}

	// 	video_writer release
	video_writer.release();

	// release rknn
	ret = rknn_destroy(ctx);
	
	// free model
	if (model_data) {
		free(model_data);
	}


	return 0;
	
}



