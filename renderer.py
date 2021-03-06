import tensorflow as tf
import mesh_renderer
from scipy.io import loadmat,savemat
import numpy as np

# define facemodel for reconstruction
class BFM():
	def __init__(self):
		model_path = './BFM/BFM_model_front.mat'
		model = loadmat(model_path)
		self.meanshape = model['meanshape'] # mean face shape 
		self.idBase = model['idBase'] # identity basis
		self.exBase = model['exBase'] # expression basis
		self.meantex = model['meantex'] # mean face texture
		self.texBase = model['texBase'] # texture basis
		self.point_buf = model['point_buf'] # adjacent face index for each vertex, starts from 1 (only used for calculating face normal)
		self.tri = model['tri'] # vertex index for each triangle face, starts from 1
		self.keypoints = np.squeeze(model['keypoints']).astype(np.int32) - 1 # 68 face landmark index, starts from 0

# compute vertex normal using one-ring neighborhood
# input: face_shape with shape [1,N,3]
# output: v_norm with shape [1,N,3]
def Compute_norm(face_shape,facemodel):

	face_id = facemodel.tri # vertex index for each triangle face, with shape [F,3], F is number of faces
	point_id = facemodel.point_buf # adjacent face index for each vertex, with shape [N,8], N is number of vertex
	shape = face_shape
	face_id = (face_id - 1).astype(np.int32)
	point_id = (point_id - 1).astype(np.int32)
	v1 = shape[:,face_id[:,0],:]
	v2 = shape[:,face_id[:,1],:]
	v3 = shape[:,face_id[:,2],:]
	e1 = v1 - v2
	e2 = v2 - v3
	face_norm = np.cross(e1,e2) # compute normal for each face
	face_norm = np.concatenate([face_norm,np.zeros([1,1,3])], axis = 1) # concat face_normal with a zero vector at the end
	v_norm = np.sum(face_norm[:,point_id,:], axis = 2) # compute vertex normal using one-ring neighborhood
	v_norm = v_norm/np.expand_dims(np.linalg.norm(v_norm,axis = 2),2) # normalize normal vectors

	return v_norm

# input: coeff with shape [1,257]
def Split_coeff(coeff):
	id_coeff = coeff[:,:80] # identity(shape) coeff of dim 80
	ex_coeff = coeff[:,80:144] # expression coeff of dim 64
	tex_coeff = coeff[:,144:224] # texture(albedo) coeff of dim 80
	angles = coeff[:,224:227] # ruler angles(x,y,z) for rotation of dim 3
	gamma = coeff[:,227:254] # lighting coeff for 3 channel SH function of dim 27
	translation = coeff[:,254:] # translation coeff of dim 3

	return id_coeff,ex_coeff,tex_coeff,angles,gamma,translation

# compute vertex texture(albedo) with tex_coeff
# input: tex_coeff with shape [1,N,3]
# output: face_texture with shape [1,N,3], RGB order, range from 0-255
def Texture_formation(tex_coeff,facemodel):

	face_texture = np.einsum('ij,aj->ai',facemodel.texBase,tex_coeff) + facemodel.meantex
	face_texture = np.reshape(face_texture,[1,-1,3])

	return face_texture

# compute rotation matrix based on 3 ruler angles
# input: angles with shape [1,3]
# output: rotation matrix with shape [1,3,3]
def Compute_rotation_matrix(angles):

	angle_x = angles[:,0][0]
	angle_y = angles[:,1][0]
	angle_z = angles[:,2][0]

	# compute rotation matrix for X,Y,Z axis respectively
	rotation_X = np.array([1.0,0,0,\
		0,np.cos(angle_x),-np.sin(angle_x),\
		0,np.sin(angle_x),np.cos(angle_x)])
	rotation_Y = np.array([np.cos(angle_y),0,np.sin(angle_y),\
		0,1,0,\
		-np.sin(angle_y),0,np.cos(angle_y)])
	rotation_Z = np.array([np.cos(angle_z),-np.sin(angle_z),0,\
		np.sin(angle_z),np.cos(angle_z),0,\
		0,0,1])

	rotation_X = np.reshape(rotation_X,[1,3,3])
	rotation_Y = np.reshape(rotation_Y,[1,3,3])
	rotation_Z = np.reshape(rotation_Z,[1,3,3])

	rotation = np.matmul(np.matmul(rotation_Z,rotation_Y),rotation_X)
	rotation = np.transpose(rotation, axes = [0,2,1])  #transpose row and column (dimension 1 and 2)

	return rotation

# compute vertex color using face_texture and SH function lighting approximation
# input: face_texture with shape [1,N,3]
# 	     norm with shape [1,N,3]
#		 gamma with shape [1,27]
# output: face_color with shape [1,N,3], RGB order, range from 0-255
#		  lighting with shape [1,N,3], color under uniform texture
def Illumination_layer(face_texture,norm,gamma):

	num_vertex = np.shape(face_texture)[1]

	init_lit = np.array([0.8,0,0,0,0,0,0,0,0])
	gamma = np.reshape(gamma,[-1,3,9])
	gamma = gamma + np.reshape(init_lit,[1,1,9])

	# parameter of 9 SH function
	a0 = np.pi 
	a1 = 2*np.pi/np.sqrt(3.0)
	a2 = 2*np.pi/np.sqrt(8.0)
	c0 = 1/np.sqrt(4*np.pi)
	c1 = np.sqrt(3.0)/np.sqrt(4*np.pi)
	c2 = 3*np.sqrt(5.0)/np.sqrt(12*np.pi)

	Y0 = np.tile(np.reshape(a0*c0,[1,1,1]),[1,num_vertex,1]) 
	Y1 = np.reshape(-a1*c1*norm[:,:,1],[1,num_vertex,1]) 
	Y2 = np.reshape(a1*c1*norm[:,:,2],[1,num_vertex,1])
	Y3 = np.reshape(-a1*c1*norm[:,:,0],[1,num_vertex,1])
	Y4 = np.reshape(a2*c2*norm[:,:,0]*norm[:,:,1],[1,num_vertex,1])
	Y5 = np.reshape(-a2*c2*norm[:,:,1]*norm[:,:,2],[1,num_vertex,1])
	Y6 = np.reshape(a2*c2*0.5/np.sqrt(3.0)*(3*np.square(norm[:,:,2])-1),[1,num_vertex,1])
	Y7 = np.reshape(-a2*c2*norm[:,:,0]*norm[:,:,2],[1,num_vertex,1])
	Y8 = np.reshape(a2*c2*0.5*(np.square(norm[:,:,0])-np.square(norm[:,:,1])),[1,num_vertex,1])

	Y = np.concatenate([Y0,Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8],axis=2)

	# Y shape:[batch,N,9].

	lit_r = np.squeeze(np.matmul(Y,np.expand_dims(gamma[:,0,:],2)),2) #[batch,N,9] * [batch,9,1] = [batch,N]
	lit_g = np.squeeze(np.matmul(Y,np.expand_dims(gamma[:,1,:],2)),2)
	lit_b = np.squeeze(np.matmul(Y,np.expand_dims(gamma[:,2,:],2)),2)

	# shape:[batch,N,3]
	face_color = np.stack([lit_r*face_texture[:,:,0],lit_g*face_texture[:,:,1],lit_b*face_texture[:,:,2]],axis = 2)
	lighting = np.stack([lit_r,lit_g,lit_b],axis = 2)*128

	return face_color,lighting

def Illumination_inv_layer(face_color,lighting):
	face_texture = np.stack([face_color[:,:,0]/lighting[:,:,0],face_color[:,:,1]/lighting[:,:,1],face_color[:,:,2]/lighting[:,:,2]],axis=2)*128
	return face_texture

# compute face shape with identity and expression coeff, based on BFM model
# input: id_coeff with shape [1,80]
#		 ex_coeff with shape [1,64]
# output: face_shape with shape [1,N,3], N is number of vertices
def Shape_formation(id_coeff,ex_coeff,facemodel):
	face_shape = np.einsum('ij,aj->ai',facemodel.idBase,id_coeff) + \
				np.einsum('ij,aj->ai',facemodel.exBase,ex_coeff) + \
				facemodel.meanshape

	face_shape = np.reshape(face_shape,[1,-1,3])
	# re-center face shape
	face_shape = face_shape - np.mean(np.reshape(facemodel.meanshape,[1,-1,3]), axis = 1, keepdims = True)

	return face_shape

def Reconstruction_for_render(coeff,facemodel):
	id_coeff,ex_coeff,tex_coeff,angles,gamma,translation = Split_coeff(coeff)
	face_shape = Shape_formation(id_coeff, ex_coeff, facemodel)
	face_texture = Texture_formation(tex_coeff, facemodel)
	face_norm = Compute_norm(face_shape,facemodel)
	rotation = Compute_rotation_matrix(angles)
	face_shape_r = np.matmul(face_shape,rotation)
	face_shape_r = face_shape_r + np.reshape(translation,[1,1,3])
	face_norm_r = np.matmul(face_norm,rotation)
	face_color,lighting = Illumination_layer(face_texture, face_norm_r, gamma)
	tri = facemodel.tri
	
	return face_shape_r,face_norm_r,face_color,tri

def Render_layer(face_shape,face_norm,face_color,facemodel,batchsize):

	camera_position = tf.constant([0,0,10.0])
	camera_lookat = tf.constant([0,0,0.0])
	camera_up = tf.constant([0,1.0,0])
	light_positions = tf.tile(tf.reshape(tf.constant([0,0,1e5]),[1,1,3]),[batchsize,1,1])
	light_intensities = tf.tile(tf.reshape(tf.constant([0.0,0.0,0.0]),[1,1,3]),[batchsize,1,1])
	ambient_color = tf.tile(tf.reshape(tf.constant([1.0,1,1]),[1,3]),[batchsize,1])

	#pdb.set_trace()
	render = mesh_renderer.mesh_renderer(face_shape,
		tf.cast(facemodel.tri-1,tf.int32),
		face_norm,
		face_color/255,
		camera_position = camera_position,
		camera_lookat = camera_lookat,
		camera_up = camera_up,
		light_positions = light_positions,
		light_intensities = light_intensities,
		image_width = 224,
		image_height = 224,
		fov_y = 12.5936,
		ambient_color = ambient_color)

	return render