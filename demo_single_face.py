import tensorflow as tf 
import numpy as np
from PIL import Image
import os
import sys
import glob
import cv2
import platform
from scipy.io import loadmat,savemat

from preprocess_img import Preprocess
from load_data import *
from face_decoder import Face3D

from MTCNN_Portable.mtcnn import MTCNN
import argparse

from renderer import *


sys.path.append("./MTCNN_Portable")

is_windows = platform.system() == "Windows"

def load_graph(graph_filename):
	with tf.gfile.GFile(graph_filename,'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	return graph_def

def demo(args):
	# input and output folder
	image_path = args.input
	save_path = args.output
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	# read BFM face model
	facemodel = BFM()
	# transfer original BFM model to our model
	if not os.path.isfile('./BFM/BFM_model_front.mat'):
		transferBFM09()

	# read standard landmarks for preprocessing images
	lm3D = load_lm3d()
	batchsize = 1

	# build reconstruction model
	with tf.Graph().as_default() as graph,tf.device('/gpu:0'):

		FaceReconstructor = Face3D()
		images = tf.placeholder(name = 'input_imgs', shape = [batchsize,224,224,3], dtype = tf.float32)
		graph_def = load_graph('network/FaceReconModel.pb')
		tf.import_graph_def(graph_def,name='resnet',input_map={'input_imgs:0': images})

		# output coefficients of R-Net (dim = 257)
		coeff = graph.get_tensor_by_name('resnet/coeff:0')

		with tf.device('/cpu:0'):
			# renderer layer
			faceshaper = tf.placeholder(name = "face_shape_r", shape = [1,35709,3], dtype = tf.float32)
			facenormr = tf.placeholder(name = "face_norm_r", shape = [1,35709,3], dtype = tf.float32)
			facecolor = tf.placeholder(name = "face_color", shape = [1,35709,3], dtype = tf.float32)
			rendered = Render_layer(faceshaper,facenormr,facecolor,facemodel,1)

			rstimg = tf.placeholder(name = 'rstimg', shape = [224,224,4], dtype=tf.uint8)
			encode_png = tf.image.encode_png(rstimg)

		# reconstructing faces
		FaceReconstructor.Reconstruction_Block(coeff,batchsize)
		face_shape = FaceReconstructor.face_shape_t
		face_texture = FaceReconstructor.face_texture
		face_color = FaceReconstructor.face_color
		landmarks_2d = FaceReconstructor.landmark_p
		recon_img = FaceReconstructor.render_imgs
		tri = FaceReconstructor.facemodel.face_buf

		# MTCNN Detector
		detector = MTCNN()
		img,lm = load_img_and_lm(image_path, detector)

		with tf.Session() as sess:
			print('reconstructing...')
			# load images and corresponding 5 facial landmarks
			
			# preprocess input image
			input_img,lm_new,transform_params,posion = Preprocess(img,lm,lm3D)

			coeff_,face_shape_,face_texture_,face_color_,landmarks_2d_,recon_img_,tri_ = sess.run([coeff,\
				face_shape,face_texture,face_color,landmarks_2d,recon_img,tri],feed_dict = {images: input_img})

			# renderer output
			face_shape_r,face_norm_r,face_color,tri = Reconstruction_for_render(coeff_,facemodel)
			final_images = sess.run(rendered, feed_dict={faceshaper: face_shape_r.astype('float32'), facenormr: face_norm_r.astype('float32'), facecolor: face_color.astype('float32')})
			result_image = final_images[0, :, :, :]
			result_image = np.clip(result_image, 0., 1.).copy(order='C')
			# save renderer output
			result_bytes = sess.run(encode_png,{rstimg: result_image*255.0})
			result_output_path = os.path.join(save_path, image_path.split(os.path.sep)[-1].replace('.png','_render.png').replace('jpg','_render.png'))
			with open(result_output_path, 'wb') as output_file:
				output_file.write(result_bytes)

			# get RGB image from RGBA
			rgb_renderer_img,mask = RGBA2RGB(result_image)
			# Paste the 3D rendered image back to the original image
			renderer_3D_input_img = np.copy(img)
			left0 = int(posion[0]*posion[4])
			right0 = int(posion[1]*posion[4])
			up0 = int(posion[2]*posion[4])
			below0 = int(posion[3]*posion[4])
			rgb_renderer_img = cv2.resize(rgb_renderer_img, (right0-left0,below0-up0))
			mask = cv2.resize(mask, (right0-left0,below0-up0))
			mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
			mask.astype('uint32')
			if left0<0:
				mask = mask[:,-left0:]
				rgb_renderer_img =rgb_renderer_img[:,-left0:]
				left0=0
			if up0<0:
				mask = mask[-up0:,:]
				rgb_renderer_img =rgb_renderer_img[-up0:,:]
				up0=0
			if right0>renderer_3D_input_img.shape[1]:
				mask = mask[:,:-(right0-renderer_3D_input_img.shape[1])]
				rgb_renderer_img =rgb_renderer_img[:,:-(right0-renderer_3D_input_img.shape[1])]
				right0=renderer_3D_input_img.shape[1]
			if below0>renderer_3D_input_img.shape[0]:
				mask = mask[:-(below0-renderer_3D_input_img.shape[0]),:]
				rgb_renderer_img =rgb_renderer_img[:-(below0-renderer_3D_input_img.shape[0]),:]
				below0=renderer_3D_input_img.shape[0]

			renderer_3D_input_img[up0:below0,left0:right0] = renderer_3D_input_img[up0:below0,left0:right0]*mask+rgb_renderer_img
			renderer_3D_input_img = cv2.cvtColor(renderer_3D_input_img, cv2.COLOR_BGR2RGB)
			cv2.imwrite(os.path.join(save_path,image_path.split(os.path.sep)[-1].replace('.png','_renderer_in_original.png').replace('jpg','_renderer_in_original.png')),renderer_3D_input_img)

			# reshape outputs
			input_img = np.squeeze(input_img)
			face_shape_ = np.squeeze(face_shape_, (0))
			face_texture_ = np.squeeze(face_texture_, (0))
			face_color_ = np.squeeze(face_color_, (0))
			landmarks_2d_ = np.squeeze(landmarks_2d_, (0))
			if not is_windows:
				recon_img_ = np.squeeze(recon_img_, (0))

			# save output files
			if not is_windows:
				savemat(os.path.join(save_path,image_path.split(os.path.sep)[-1].replace('.png','.mat').replace('jpg','mat')),{'cropped_img':input_img[:,:,::-1],'recon_img':recon_img_,'coeff':coeff_,\
					'face_shape':face_shape_,'face_texture':face_texture_,'face_color':face_color_,'lm_68p':landmarks_2d_,'lm_5p':lm_new})
			save_obj(os.path.join(save_path,image_path.split(os.path.sep)[-1].replace('.png','_mesh.obj').replace('jpg','_mesh.obj')),face_shape_,tri_,np.clip(face_color_,0,255)/255) # 3D reconstruction face (in canonical view)

# load input images and corresponding 5 landmarks
def load_img_and_lm(img_path, detector):
	print("Reading image")
	image = Image.open(img_path)
	if img_path.split('.')[-1]=='png':
		image = image.convert("RGB")
	img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
	print("Detect 5 key point")
	face = detector.detect_faces(img)[0]
	left_eye = face["keypoints"]["left_eye"]
	right_eye = face["keypoints"]["right_eye"]
	nose = face["keypoints"]["nose"]
	mouth_left = face["keypoints"]["mouth_left"]
	mouth_right = face["keypoints"]["mouth_right"]
	lm = np.array([[left_eye[0], left_eye[1]],
				[right_eye[0], right_eye[1]],
				[nose[0], nose[1]],
				[mouth_left[0], mouth_left[1]],
				[mouth_right[0], mouth_right[1]]])
	return image,lm

# get RGB image from RGBA
def RGBA2RGB(img):
	img = img[:,:,0:3]
	img = img*255
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, mask = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)
	
	return img, (255-mask)/255


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', type=str, default='./MTCNN_Portable/jx.jpg',help='Path of input image')
	parser.add_argument('--output', type=str, default='./output',
						help='Path of output fold')
	arguments = parser.parse_args()
	demo(arguments)
