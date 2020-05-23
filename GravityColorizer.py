#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script performs image operations on pyopencl in combination with PIL
#
# loosely based on the code of: https://gist.github.com/likr/3735779

import pyopencl as cl
from pyopencl import array
import numpy as np
from PIL import Image
import yaml
import argparse
from copy import deepcopy

#RGBA Color Format
R = 0
G = 1
B = 2
A = 3

#2D Position
X = 0
Y = 1

def Interpolate(x0, x1, y0, y1, x):
	return y0 + (x - x0)*(y1 - y0)/(x1 - x0)

def ParseCommandLineArgs():
	path_def = './GravityColorizer_keyframes.yaml'
	out_def = 'result'
	
	prog_desc = ('Given a path to a yaml file that denotes keyframes (see template) '
		'create a picture that colors each pixel for each frame '
		'based on where an asteroid in the pixel\'s position would end up. '
		'REQUIRES: PyOpenCL, yaml, and PIL ')
	path_help = ('Path to the keyframes yaml file, '
		'Default: ' + path_def)
	out_help = ('Name of the image file to output. '
		'The name should exclude the extension. Image format will be gif. '
		'Default: ' + out_def)
	
	parser = argparse.ArgumentParser(description = prog_desc)
	parser.add_argument('--path', '-p', type = str, help = path_help)
	parser.add_argument('--out',  '-o', type = str, help = out_help)
	
	parser.set_defaults(path = path_def, out = out_def)
	
	args = parser.parse_args()
	
	return args
	
def ParseKeyframeFile(keyframe_path):
	num_frames = 0
	frame_width = 0
	frame_height = 0
	keyfrm_dict = {}
	yaml_obj = None
	
	try:
		with open(keyframe_path) as f:
			yaml_obj = yaml.load(f, yaml.SafeLoader)
	except Exception as err:
		print('Failed to get info from path "' + keyframe_path + '". Error message: "' + str(err) + '"')
		exit()
	
	keyfrm_dict = deepcopy(yaml_obj)
	
	return keyfrm_dict

def InitializeGravityColorizer():
	#Obtain an OpenCL platform (TODO: Give users the ability to choose which GPU to use).
	#NOTE: last index is used here because at the time of writing author's platforms are: [CPU, GPU]
	platform = cl.get_platforms()[-1]
	
	#Obtain a device id for at least one device (accelerator).
	device = platform.get_devices()[0]
	
	#Create a context for the selected device.
	ctx = cl.Context([device])
	#NOTE: This line may or may not also be a valid way of getting the context
	#ctx = cl.create_some_context()
	
	#Get the queue so that we may add a task (involving the program below) to it.
	queue = cl.CommandQueue(ctx)

	# load and build OpenCL function
	#NOTE: According to https://stackoverflow.com/questions/25114580/opencl-pass-image2d-t-twice-to-get-both-read-and-write-from-kernel
	#  Having an image be read-write violates CL 1.x standard. May not be optimal for CL 2.x standard
	prg = cl.Program(ctx, '''//CL//
	__kernel void GravityColorizer(
		write_only image2d_t img_arr,
		uint n,
		uint dt,
		uint num_pmos,
		__global const uint4 *pmos_c,
		__global const float2 *pmos_p,
		__global const float *pmos_m,
		__global const float *pmos_r)
	{
		//Parameters benefiting from scalability
		int2 pos = (int2)(get_global_id(0), get_global_id(1));
		uint4 pix = {0, 0, 0, 255}; //Black pixel with alpha value to opaque.
		
		//Asteroid statistics (can't be altered by user currently)
		float2 ast_p = {(float)pos.x + 0.5, (float)pos.y + 0.5};
		float2 ast_v = {0.0, 0.0};
		float2 ast_a = {0.0, 0.0};
		float ast_m = 1.0;
		float ast_r = 0.0; //asteroid has negligible radius.
		
		//Gravitational constant to six significant figures
		const float G = 6.67408 * pow(10.0, -11);
		
		for(uint i = 0; i < n; i++)
		{
			float2 f = {0.0, 0.0};
			
			//Compute the force acted on the asteroid 
			for(uint pmo_idx = 0; pmo_idx < num_pmos; pmo_idx++)
			{
				float dx = pmos_p[pmo_idx].x - ast_p.x;
				float dy = pmos_p[pmo_idx].y - ast_p.y;
				float d = sqrt(dx*dx + dy*dy);
				
				if(d < pmos_r[pmo_idx] + ast_r)
				{
					//If asteroid is within an pmo, return the pmo's color, darkened for each iteration it's taken.
					float factor = (float)(n - i) / (float)(n);
					pix.x = (uchar)((float)pmos_c[pmo_idx].x * factor);
					pix.y = (uchar)((float)pmos_c[pmo_idx].y * factor);
					pix.z = (uchar)((float)pmos_c[pmo_idx].z * factor);
					write_imageui(img_arr, pos, pix);
					return;
				}
				else
				{
					//Vectorized form of Newton's Universal Law of Gravitation:
					//F = G * m1 * m2 * (ud)/(d^2)
					//Where ud is the unit vector of the distance between m1 and m2 (dx/d, dy/d)
					f.x += (G * (pmos_m[pmo_idx] * ast_m) * dx) / (d * d * d);
					f.y += (G * (pmos_m[pmo_idx] * ast_m) * dy) / (d * d * d);
				}
			}
			
			//Advancement Step
			ast_a.x = f.x / ast_m;
			ast_a.y = f.y / ast_m;
			ast_v.x += ast_a.x * dt;
			ast_v.y += ast_a.y * dt;
			ast_p.x += ast_v.x * dt;
			ast_p.y += ast_v.y * dt;
		}
		
		//Asteroid did not hit any PMO in n steps. Return default pixel.
		write_imageui(img_arr, pos, pix);
	}
	''').build()
	
	return (ctx, queue, prg)

def GetValueFromMethod(method, t):
	value = 0
	
	if type(method) == int or type(method) == float:
		value = method
	else:
		#Assume it's a string and use shady and unsafe eval() function.
		value = eval(method)
	
	return value

def CreateArrayFromMaybeKeyFrames(maybe_keyfrm_dict, num_frames):
	arr = []
	
	if type(maybe_keyfrm_dict) != dict:
		#There are no keyframes. Value is either static or dynamic based off single function.
		method = maybe_keyfrm_dict
		for t in range(num_frames):
			arr.append(GetValueFromMethod(method, t))
	else:
		keyfrm_dict = maybe_keyfrm_dict
		keyfrm_idx = 0
		keyfrm_key_arr = sorted(keyfrm_dict.keys())
		for t in range(num_frames):
			#Update the key frame index to match the current frame that we're looking at
			if keyfrm_idx < len(keyfrm_dict) - 1 and keyfrm_key_arr[keyfrm_idx + 1] <= t:
				keyfrm_idx += 1
			
			#Obtain the value per frame index from the keyfrm_dict.
			#Assume it is a string and use the shady and unsafe eval() function.
			arr.append(GetValueFromMethod(keyfrm_dict[keyfrm_key_arr[keyfrm_idx]], t))

	return arr

def ProcessKeyFrames(keyfrm_dict, out_filename):
	print('Setting up...')
	
	(ctx, queue, prg) = InitializeGravityColorizer()
	img_arr = []
	percent = 1
	
	#Static across all frames
	num_frames = np.uint32(keyfrm_dict['num_frames'])
	frm_idx_arr = list(range(num_frames))
	frame_width = np.uint32(keyfrm_dict['frame_width'])
	frame_height = np.uint32(keyfrm_dict['frame_height'])
	num_pmos = np.uint32(len(keyfrm_dict['pmos'].keys()))
	
	n_arr = CreateArrayFromMaybeKeyFrames(keyfrm_dict['n'], num_frames)
	dt_arr = CreateArrayFromMaybeKeyFrames(keyfrm_dict['dt'], num_frames)
	
	pmos_c_arr_dict = {}
	pmos_p_arr_dict = {}
	pmos_m_arr_dict = {}
	pmos_r_arr_dict = {}
	for pmos_key in keyfrm_dict['pmos'].keys():
		pmos_cr_arr = CreateArrayFromMaybeKeyFrames(keyfrm_dict['pmos'][pmos_key]['cr'], num_frames)
		pmos_cg_arr = CreateArrayFromMaybeKeyFrames(keyfrm_dict['pmos'][pmos_key]['cg'], num_frames)
		pmos_cb_arr = CreateArrayFromMaybeKeyFrames(keyfrm_dict['pmos'][pmos_key]['cb'], num_frames)
		pmos_px_arr = CreateArrayFromMaybeKeyFrames(keyfrm_dict['pmos'][pmos_key]['px'], num_frames)
		pmos_py_arr = CreateArrayFromMaybeKeyFrames(keyfrm_dict['pmos'][pmos_key]['py'], num_frames)
		pmos_m_arr = CreateArrayFromMaybeKeyFrames(keyfrm_dict['pmos'][pmos_key]['m'], num_frames)
		pmos_r_arr = CreateArrayFromMaybeKeyFrames(keyfrm_dict['pmos'][pmos_key]['r'], num_frames)
		
		#Add to dict
		pmos_c_arr_dict[pmos_key] = [pmos_cr_arr, pmos_cg_arr, pmos_cb_arr]
		pmos_p_arr_dict[pmos_key] = [pmos_px_arr, pmos_py_arr]
		pmos_m_arr_dict[pmos_key] = pmos_m_arr
		pmos_r_arr_dict[pmos_key] = pmos_r_arr
	
	#PMOs Arrays Creation (Note: Assumes that the first frame has all of the pmos listed)
	pmos_c = np.zeros((1, num_pmos), cl.array.vec.uint4)
	pmos_p = np.zeros((1, num_pmos), cl.array.vec.float2)
	pmos_m = np.zeros((1, num_pmos), dtype=np.float32)
	pmos_r = np.zeros((1, num_pmos), dtype=np.float32)
	
	print('Setup Complete. Creating Visualization.')
	for frm_idx in range(num_frames):
		n = np.uint32(n_arr[frm_idx])
		dt = np.uint32(dt_arr[frm_idx])
		
		for pmos_idx, pmos_key in enumerate(sorted(keyfrm_dict['pmos'].keys())):
			pmos_c[0, pmos_idx] = (int(np.rint(pmos_c_arr_dict[pmos_key][R][frm_idx])), int(np.rint(pmos_c_arr_dict[pmos_key][G][frm_idx])), int(np.rint(pmos_c_arr_dict[pmos_key][B][frm_idx])), 255)
			pmos_p[0, pmos_idx] = (pmos_p_arr_dict[pmos_key][X][frm_idx], pmos_p_arr_dict[pmos_key][Y][frm_idx])
			pmos_m[0, pmos_idx] = (pmos_m_arr_dict[pmos_key][frm_idx])
			pmos_r[0, pmos_idx] = (pmos_r_arr_dict[pmos_key][frm_idx])
		
		#Convert vector inputs to OpenCL buffers
		#These buffers must be recreated at any time the host buffer changes.
		pmos_c_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=pmos_c)
		pmos_p_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=pmos_p)
		pmos_m_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=pmos_m)
		pmos_r_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=pmos_r)
		
		#Create source image
		#This example code only works with RGBA images
		img_np_arr = np.zeros((frame_width, frame_height, 4), dtype=np.uint8)
	
		#Build destination OpenCL Image
		fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
		img_buf = cl.Image(ctx, cl.mem_flags.WRITE_ONLY, fmt, shape=(frame_width, frame_height))
		
		#Execute OpenCL function
		prg.GravityColorizer(queue, (frame_width, frame_height), None, img_buf, n, dt, num_pmos, pmos_c_buf, pmos_p_buf, pmos_m_buf, pmos_r_buf)
		
		#Copy result back to host
		cl.enqueue_copy(queue, img_np_arr, img_buf, origin=(0, 0), region=(frame_width, frame_height))
		
		#Convert the np array into a frame and add it to the frame array
		img = Image.fromarray(img_np_arr)
		img_arr.append(img)
		
		#Display percent done
		if float(frm_idx)/float(num_frames) > float(percent)/100:
				print(str(percent) + '% done')
				percent += 1
	
	print('100% done')
	
	#Save the img_arr as a gif
	#duration=33 ==> ~30fps
	#loop=0 ==> gif will repeat after finishing.
	img_arr[0].save(out_filename + '.gif', format='GIF', loop=0, duration=33, save_all=True, append_images=img_arr[1:])
	
	#NOTE: Gif files only use 256 colors per frame, and this program will frequently exceed that even if the planet is only one color.
	#  The result is a large file with extremely noticeable dithering or "rings" around each pmo.
	#  The reason for this occurring is most likely due to the finely detailed gradients that this program creates for certain values of n and dt.
	#  Dithering can be outright eliminated by using programs such as gifski on an array of png files, like so:
	#Uncomment for png array:
	##for i in range(len(img_arr)):
	##	img_arr[i].save(out_filename + '_' + str(i) + '.png', format='PNG')
	#NOTE: gifski is open source and uses the AGPL 3 license.

def Main():
	args = ParseCommandLineArgs()
	keyfrm_dict = ParseKeyframeFile(args.path)
	ProcessKeyFrames(keyfrm_dict, args.out)

if __name__ == "__main__":
	Main()