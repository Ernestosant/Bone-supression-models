import gradio as gr
from cv2 import imread, imwrite, COLOR_BGR2GRAY, COLOR_GRAY2BGR, equalizeHist, cvtColor, resize, INTER_LINEAR
from fastai.learner import load_learner
from numpy import asarray, float32, uint8, zeros, reshape

from gradio.components import Textbox

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from gradio.components import Dropdown

def resize(input_image, height, width):
  m, n, c = input_image.shape
  if(m>255 or n >255):
    input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) 
  return input_image

# Normalizing the images to [-1, 1]
def normalize(input_image):
  input_image = (input_image / 127.5) - 1

  return input_image

def histeq (img):
    imgray = cvtColor(img,COLOR_BGR2GRAY)
    imhist = equalizeHist(imgray)
    imhist_rgb = cvtColor(imhist,COLOR_GRAY2BGR)
    return imhist_rgb
def get_y():
  return True
# unet inference
def get_mask(model, img):  
  if len(img.shape)<3:
    img1 = cvtColor(img,COLOR_GRAY2BGR)
    img1 = histeq(img1)
    imwrite('img.tif',img1)
  img = histeq(img)
  imwrite('img.tif',img)
  #load inference image
  dl = model.dls.test_dl("img.tif")
  # get the prediction
  preds =model.get_preds(dl=dl)
  print(preds[0][0].shape)
  mso_img = preds[0][0].permute(1, 2, 0)

  print(mso_img.shape)
  mso_img = uint8((((asarray(mso_img))+3)/6)*255)
  
  return mso_img

def get_mask_by_steeps(model, img, steeps=2):
  im1=get_mask(model,img)
  if(steeps>1):
    for i in range(steeps-1):
      im1=get_mask(model, im1)
      im1=histeq(im1)
    return im1


# GAN inference
from tensorflow.python import training
def get_pred(model, image, steeps=1):
  image = histeq(image)
  img=tf.convert_to_tensor(image, tf.float32)
  m,n,c = image.shape
  if(m>256 or n>255):
    img= resize(img,256, 256)
  img=normalize(img)
  inp=img[:,:,0:3]
  print(inp.shape)
  gen_output = model(inp[tf.newaxis, ...], training=True)
  pred = gen_output[0, ...]
  pred=pred*0.5+0.5
  pred = np.asarray(pred*255, np.uint8)
  pred = histeq(pred)
  if(steeps>1):
    for i in range(steeps): 
      im = pred      
      im = tf.convert_to_tensor(im,tf.float32)
      im = normalize(im)
      gen_output = model(im[tf.newaxis, ...], training=True)
      pred = gen_output[0, ...]
      pred=pred*0.5+0.5
      pred = np.asarray(pred*255, np.uint8)
      pred = histeq(pred)
  return pred


def load_custom_model(model_name, path_model):
  if(model_name == "unet_resnet50"):
    model = load_learner(path_model)
  elif(model_name=='GAN_pix2pix'):
    model =tf.keras.models.load_model(path_model)
  return  model

def get_custom_pred(model, model_name, img):  
  if(model_name == "unet_resnet50"):
    predf = get_mask_by_steeps(model, img, 2)
  elif(model_name=='GAN_pix2pix'):
    predf =get_pred(model, img, steeps=2)
  return predf


def get_BSIMG(img, model_name, path_model):
  pred = get_custom_pred(load_custom_model(model_name, path_model), model_name, img)
  return pred

  # Creating gradio interface
model_select = Dropdown(
    choices=["unet_resnet50", "GAN_pix2pix"], 
    default="unet_resnet50",
    label="Select model"    
)
path_model = Textbox(label='Path model')

demo = gr.Interface(fn = get_BSIMG, inputs=['image', model_select, path_model], outputs='image')
demo.launch()
