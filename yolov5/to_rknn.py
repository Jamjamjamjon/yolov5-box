import yaml
from rknn.api import RKNN
import cv2
import os
import argparse



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cali', type=str, default='yolov5/projects/coco/calibration.txt')
    parser.add_argument('--onnx-path', type=str, default='yolov5/projects/coco/weights/yolov5s.onnx')
    
    parser.add_argument('--batch-size', type=int, default=50, help='batch size')
    parser.add_argument('--epochs', type=int, default=-1, help='epoches')
    parser.add_argument('--optimization_level', type=int, default=3, help='opt level')


    parser.add_argument('--infer',  action='store_true', help="run inference")
    parser.add_argument('--eval-perf',  action='store_true', help="run eval perfermance")
    parser.add_argument('--test_img-path', type=str, default='../data/images/bus.jpg')
    #parser.add_argument('--rknn-path', type=str)



    args = parser.parse_args()
    return args


def main(args):


 
    model_type = "onnx"
    print('model_type is {}'.format(model_type))

    rknn = RKNN(verbose=True)

    print('--> config model')

    # --rknn_mode
    rknn.config(  channel_mean_value = '0 0 0 255', # 123.675 116.28 103.53 58.395 # 0 0 0 255 # 
                  reorder_channel = '0 1 2', # '0 1 2' '2 1 0'
                  # need_horizontal_merge = True,
                  batch_size = args.batch_size,
                  epochs = args.epochs,
                  # target_platform = ['rk3399pro'],
                  quantized_dtype = 'asymmetric_quantized-u8', # asymmetric_quantized-u8,dynamic_fixed_point-8,dynamic_fixed_point-16
                  optimization_level = args.optimization_level)


    # --add_image_preprocess_layer
    # rknn.config(  
    #               # channel_mean_value = '0 0 0 255', # 123.675 116.28 103.53 58.395 # 0 0 0 255 # 
    #               # reorder_channel = '0 1 2', # '0 1 2' '2 1 0'
    #               need_horizontal_merge = True,
    #               batch_size = args.batch_size,
    #               epochs = args.epochs,
    #               target_platform = ['rk3399pro'],
    #               quantized_dtype = 'asymmetric_quantized-u8', # asymmetric_quantized-u8,dynamic_fixed_point-8,dynamic_fixed_point-16
    #               optimization_level = 1)
    print('done')


    print('--> Loading model')
    ret = rknn.load_onnx(args.onnx_path)
    if ret != 0:
        print('Load mobilenet_v2 failed! Ret = {}'.format(ret))
        exit(ret)
    print('done')

 
    if model_type != 'rknn':
        print('--> Building model')
        ret = rknn.build(  do_quantization = True,
                            dataset = args.cali,
                            pre_compile = False)
        if ret != 0:
            print('Build failed!')
            exit(ret)
    else:
        print('--> skip Building model step, cause the model is already rknn')


 
    print('--> Export RKNN model')
    saveout = args.onnx_path.replace(".onnx", ".rknn")
    print(f"saveout: {saveout}")
    
    ret = rknn.export_rknn(saveout)
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    else:
        print('done')


    if args.infer or args.eval_perf:
        print('--> Init runtime environment')
        res = rknn.init_runtime(  target = None,
							  device_id = None,
							  perf_debug = False,
							  eval_mem = False,
							  async_mode= False,)        
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)

        print('--> load img')
        img = cv2.imread(args.test_img_path)
        print('img shape is {}'.format(img.shape))
        inputs = [img]


        if args.infer is True:
            print('--> Running model')

            outputs = rknn.inference(inputs)
            print('len of output {}'.format(len(outputs)))
            print('outputs[0] shape is {}'.format(outputs[0].shape))
            print(outputs[0][0][0:2])
        else:
            print('--> skip inference')


        if args.eval_perf is True:
            print('--> Begin evaluate model performance')
            perf_results = rknn.eval_perf(inputs=[img])
        else:
            print('--> skip eval_perf')
    else:
        print('--> skip inference')
        print('--> skip eval_perf')

if __name__ == '__main__':

    args = parse_args()
    main(args)
