import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
import rich

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


import models

from models.common import Conv
from models.experimental import attempt_load
from models.yolo import Detect
from utils.activations import SiLU
from utils.datasets import LoadImages
from utils.general import colorstr, check_dataset, check_img_size, check_requirements, file_size, print_args, \
    set_logging, url2file
from utils.torch_utils import select_device


# ONNX
def export_onnx(model, im, file, opset, train, dynamic, simplify, imgsz=640):
    # YOLOv5 ONNX export
    try:
        check_requirements(('onnx',))
        import onnx

        rich.print(f"[bold blue]ONNX Version: [bold white]{onnx.__version__}")

        f = file.with_name(f"{file.stem}_{str(imgsz)}.onnx")
        # f = file.with_suffix('.onnx')


        torch.onnx.export(model, im, f, verbose=False, opset_version=opset,
                          training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=not train,
                          input_names=['images'],
                          output_names=['output'],
                          dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
                                        'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
                                        } if dynamic else None)

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # print(onnx.helper.printable_graph(model_onnx.graph))  # print

        # Simplify
        if simplify:
            try:
                check_requirements(('onnx-simplifier',))
                import onnxsim

                rich.print(f"[bold blue]ONNX Simplifier Version: [bold white]{onnxsim.__version__}")
                model_onnx, check = onnxsim.simplify(
                    model_onnx,
                    dynamic_input_shape=dynamic,
                    input_shapes={'images': list(im.shape)} if dynamic else None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                print(f'{prefix} simplifier failure: {e}')
        rich.print(f"[blue bold]ONNX export success, saved as: [bold white]{f} ({file_size(f):.1f} MB)")
        # rich.print(f"[gold1 bold]ONNX Runtime run --dynamic ONNX model inference with: 'python detect.py --weights {f}'")
    except Exception as e:
        rich.print(f"[bold red]export failure: {e}")


@torch.no_grad()
def run(data=ROOT / 'data/coco128.yaml',  # 'dataset.yaml path'
        weights=ROOT / 'yolov5s.pt',  # weights path
        imgsz=(640, 640),  # image (height, width)
        batch_size=1,  # batch size
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        include=('onnx'),  # include formats
        half=False,  # FP16 half-precision export
        inplace=False,  # set YOLOv5 Detect() inplace=True
        train=False,  # model.train() mode
        dynamic=False,  # ONNX/TF: dynamic axes
        simplify=False,  # ONNX: simplify model
        opset=11,  # ONNX: opset version

        permute="yes",
        spp_optimize="yes",
        ):
    t = time.time()
    include = [x.lower() for x in include]
    tf_exports = list(x in include for x in ('saved_model', 'pb', 'tflite', 'tfjs'))  # TensorFlow exports
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    onnx_size_suffix = imgsz if len(imgsz) == 1 else imgsz[0]   # for onnx size suffix

    file = Path(url2file(weights) if str(weights).startswith(('http:/', 'https:/')) else weights)

    # Load PyTorch model
    device = select_device(device)
    assert not (device.type == 'cpu' and half), '--half only compatible with GPU export, i.e. use --device 0'
    model = attempt_load(weights, map_location=device, inplace=True, fuse=True)  # load FP32 model
    nc, names = model.nc, model.names  # number of classes, class names

    # Input
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz = [check_img_size(x, gs) for x in imgsz]  # verify img_size are gs-multiples
    im = torch.zeros(batch_size, 3, *imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection

    # Update model
    if half:
        im, model = im.half(), model.half()  # to FP16
    model.train() if train else model.eval()  # training mode = no Detect() layer grid construction


    #--------------------------
    # optimize info @jamjon
    #--------------------------
    rich.print("[bold blue]Export Info:")
    if spp_optimize in ["yes", "y", "ok"]:
        rich.print("[gold1 bold]  ==> SPP/SPPF optimize, use multi 3x3 to replace [5, 9, 13] MaxPool2d")
        rich.print("  ==> use param [bold red]--spp_optimize no[/bold red] to cancel.")
    if permute in ["yes", "y", "ok"]:
        rich.print(f"[gold1 bold]  ==> ONNX export with permute layer, shape like: [1, 3, 80, 80, 85]")
        rich.print("  ==> use param [bold red]--permute no[/bold red] to cancel.")
    elif permute in ["no", "NO", "n"]:
        rich.print(f"[gold1 bold]  ==> ONNX export with no permute layer, shape like: [1, 255, 80, 80]")
        rich.print("  ==> use param [bold red]--permute yes[/bold red] to cancel.")

    #------------------------------

    for k, m in model.named_modules():
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif isinstance(m, Detect):
            m.inplace = inplace
            m.onnx_dynamic = dynamic
            # m.forward = m.forward_export  # assign forward (optional)

        # ----------------------------
        #       customized @jamjon
        # ----------------------------
        if spp_optimize in ["yes", "y", "ok"]:
            # SPP() : [5,9,13] maxpool ==> multi 3x3 size 
            if isinstance(m, models.common.SPP):  # assign export-friendly activations
                m.m[0] = nn.Sequential(*[nn.MaxPool2d(kernel_size=3, stride=1, padding=1) for i in range(2)])
                m.m[1] = nn.Sequential(*[nn.MaxPool2d(kernel_size=3, stride=1, padding=1) for i in range(4)])
                m.m[2] = nn.Sequential(*[nn.MaxPool2d(kernel_size=3, stride=1, padding=1) for i in range(6)])

            # SPPF() @jamjon 21.10.20 
            if isinstance(m, models.common.SPPF): 
                m.m = nn.Sequential(*[nn.MaxPool2d(kernel_size=3, stride=1, padding=1) for i in range(2)])

        # if opt.no_permute:
        if permute in ["N", "no", "NO", "n"]:
            model.model[-1].permute = False


    for _ in range(2):
        y = model(im)  # dry runs
    rich.print(f"[bold blue]PyTorch: {file} ({file_size(file):.1f} MB)")

    # Exports
    if 'onnx' in include:
        export_onnx(model, im, file, opset, train, dynamic, simplify, onnx_size_suffix)

    # Finish
    # rich.print(f"[bold green]Export complete ({time.time() - t:.2f}s)")
    rich.print(f"[bold blue]Results saved: [bold white]{file.parent.resolve()}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='weights path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='image (h, w)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set YOLOv5 Detect() inplace=True')
    parser.add_argument('--train', action='store_true', help='model.train() mode')

    parser.add_argument('--dynamic', action='store_true', help='ONNX/TF: dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=11, help='ONNX: opset version')
    parser.add_argument('--include', nargs='+', default=['onnx'], help='available formats are (torchscript, onnx, coreml, saved_model, pb, tflite, tfjs)')
    
    parser.add_argument('--permute', type=str, default='yes', help='export ONNX without permute layer')   
    parser.add_argument('--spp-optimize', type=str, default='yes', help='optimize spp/sppf')   


    opt = parser.parse_args()
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    set_logging()
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
