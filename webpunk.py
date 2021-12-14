import streamlit as st 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
model = load_model('./decodervaenosigmoid_punk.h5',compile=False)
model2 = load_model('./decoderbetavae1000_punk.h5',compile=False)

st.set_page_config(layout="wide")
r = np.zeros((1,576))

image = Image.open('./punks.png')
# model = load_model('./decoder576final_punk.h5',compile=False)
# vae = load_model('./vae576_punk.h5',compile=False)
# enc = load_model('./encoder576final_punk.h5',compile = False)
col11,col12 = st.columns([1,3])
def get_punk(id):
    col = int((id-1)%100)
    row = int((id-1)/100)
        
    x= col*24
    y = row*24
    x2 = 24+x
    y2= 24+y
    
    img = image.crop((x,y,x2,y2))
    img2 = img.resize((240,240))
    # img2.show()
    return img
ids = 1
with col11:
    st.header('Crypto_punks  ')
    ids = st.number_input('punk_number',1,10000,value = 100,step=1)
    img = get_punk(ids)
with col12:
    st.image(img,width = 200)
    st.text('#'+str(ids))
    
    
    
    

    # return image.crop((x,y,x2,y2))

# st.image(get_punk(200),width=200)
# latent = enc.predict(np.expand_dims(get_punk(1),axis=0))
# remodel_img = model.predict(np.random.randn(10000,576))

    
col1 , col2,col3 = st.columns(3)

with col1:
    st.subheader('Random Vector')
    if (st.button('generate punk')):
        r = np.random.randn(1,576)
    
    
with col2:
    st.subheader('VAE')
    i = model.predict(r)
    i = i[0]
    i = np.clip(i,0,1)    
    st.image(i,width = 200)
    
with col3:
    st.subheader('Beta - VAE')
    i = model2.predict(r)
    i = i[0]
    i = np.clip(i,0,1)
    
    st.image(i,width = 200)
