import torch
from models.experimental import attempt_load
import argparse

def parse_opt():
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights',type=str, default="", help='model path(s)')
	opt = parser.parse_args()	
	return opt	

opt = parse_opt()

model = attempt_load(opt.weights, map_location=torch.device('cpu'))
m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]
# print(m.anchors)
# print(m.stride)
anchors = m.anchors * m.stride.to(m.anchors.device).view(-1, 1, 1)  # current anchors
print(anchors)