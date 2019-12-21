import sys, os, ast
import numpy as np
import cv2
import yaml


def shift_patch(patchpath, bgpath, patchsize, bgsize, stride, outdir, expname):
	
	if not os.path.isdir(outdir):
		os.mkdir(outdir)

	patch = cv2.imread(patchpath, cv2.IMREAD_UNCHANGED)
	bg = cv2.imread(bgpath, cv2.IMREAD_COLOR)

	tup_patchsize = ast.literal_eval(patchsize)
	tup_bgsize = ast.literal_eval(bgsize)

	patch = cv2.resize(patch, tup_patchsize)
	bg = cv2.resize(bg, tup_bgsize)

	a, b, _ = patch.shape
	m, n, _ = bg.shape

	counter = 0
	for i in range(0, m-a+1, stride):
		for j in range(0, n-b+1, stride):
			out = bg.copy()

			p_alpha = patch[:,:,3] / 255.0
			out[i:i+a,j:j+b,0] = (1 - p_alpha) * bg[i:i+a,j:j+b,0] + \
				p_alpha * patch[:,:,0]
			out[i:i+a,j:j+b,1] = (1 - p_alpha) * bg[i:i+a,j:j+b,1] + \
				p_alpha * patch[:,:,1]
			out[i:i+a,j:j+b,2] = (1 - p_alpha) * bg[i:i+a,j:j+b,2] + \
				p_alpha * patch[:,:,2]

			out_name = "{}_{}_{}.png".format(i, j, expname)
			out_path = os.path.join(outdir, out_name)

			cv2.imwrite(out_path, out)
			counter += 1

	return counter

def shift_crop(imgpath, imgsize, cropsize, stride, outdir, expname):
	
	if not os.path.isdir(outdir):
		os.mkdir(outdir)

	img = cv2.imread(imgpath, cv2.IMREAD_COLOR)

	tup_imgsize = ast.literal_eval(imgsize)
	tup_cropsize = ast.literal_eval(cropsize)

	img = cv2.resize(img, tup_imgsize)

	a, b = tup_cropsize
	m, n = tup_imgsize

	counter = 0
	for i in range(0, m-a+1, stride):
		for j in range(0, n-b+1, stride):
			out = img[i:i+a, j:j+b, :]

			out_name = "{}_{}_{}.png".format(i, j, expname)
			out_path = os.path.join(outdir, out_name)

			cv2.imwrite(out_path, out)
			counter += 1

	return counter

def usage():
	print("Usage: python sweep_patch.py sweep_config")
	sys.exit(0)

if __name__ == "__main__":

	if len(sys.argv) != 2:
		usage()

	# Load config
	with open(sys.argv[1], 'r') as f:
		config = yaml.load(f, Loader=yaml.FullLoader)

	if config['shiftpatch'] == config['shiftcrop']:
		print("Error: Only one data generation method must be selected.")
		sys.exit(1)

	if config['shiftpatch']:
		count = shift_patch(
			patchpath=config['patch_config']['patch_filepath'],
			bgpath=config['patch_config']['bg_filepath'],
			patchsize=config['patch_config']['patch_size'],
			bgsize=config['patch_config']['bg_size'],
			stride=config['patch_config']['stride'],
			outdir=config['out_dir'],
			expname=config['name'])

	if config['shiftcrop']:
		count = shift_crop(
			imgpath=config['crop_config']['img_filepath'],
			imgsize=config['crop_config']['img_size'],
			cropsize=config['crop_config']['crop_size'],
			stride=config['crop_config']['stride'],
			outdir=config['out_dir'],
			expname=config['name'])

	print("{} images generated.".format(count))

