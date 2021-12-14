import streamlit as st 
import math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle as pkl 
import seaborn as sns
import plotly.figure_factory as ff
st.set_page_config(layout="wide")

padding = 3
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        
    }} </style> """, unsafe_allow_html=True)
st.sidebar.header('    Crypto_punks')

page = st.sidebar.selectbox(label='models',options=(
    "Home",
    "VAE",
    "Beta-VAE",
    "Beta-VAE 2 ",
    "Latent Space properties"
))
                        
if (page =="Home"):
    
    model = load_model('./decodervaenosigmoid_punk.h5',compile=False)
    model2 = load_model('./decoderbetavae1000_punk.h5',compile=False)

    r = np.zeros((1,576))

    image = Image.open('./punks.png')
    # model = load_model('./decoder576final_punk.h5',compile=False)
    # vae = load_model('./vae576_punk.h5',compile=False)
    # enc = load_model('./encoder576final_punk.h5',compile = False)
    col11,col12 = st.columns([1,1])
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


elif(page =="VAE"):
    
    model = load_model('./decodervaenosigmoid_punk.h5',compile = False)



    preds = model.predict(np.random.randn(400,576))

    def plot_images(rows, cols, images):
        grid = np.zeros(shape=(rows*24, cols*24,3))
        for row in range(rows):
            for col in range(cols):
                grid[row*24:(row+1)*24, col*24:(col+1)*24, :] = images[row*cols + col]

        return grid

    st.header("Randomly generated 400 images with VAE ")
    n= st.number_input("number of images to be generated",25,400,100)
    number =int( math.sqrt(n))
    img = plot_images(number,number,preds)
    img = np.clip(img,0,1)
    st.image(img,width = 400)
    st.text('These are '+str(n)+' Images created using Variational Auto Encoder with latent-space of 576.')



elif(page =="Beta-VAE"):
    
    model = load_model('./decoderbetavae1000_punk.h5',compile = False)
    preds = model.predict(np.random.randn(400,576))

    def plot_images(rows, cols, images):
        grid = np.zeros(shape=(rows*24, cols*24,3))
        for row in range(rows):
            for col in range(cols):
                grid[row*24:(row+1)*24, col*24:(col+1)*24, :] = images[row*cols + col]

        return grid

    st.header("Randomly generated images with Beta-VAE ")
    n= st.number_input("number of images to be generated",25,400,100)
    number =int( math.sqrt(n))
    img = plot_images(number,number,preds)
    img = np.clip(img,0,1)
    st.image(img,width = 400)
    st.text(str(n)+' Images created using Beta Variational Auto Encoder with latent-space of 576.')
    st.text('In this proces the Beta is considered as 0.02 to be in scale with reconstruction loss .')
    st.text('at the same time have an emphasis on KL-Loss')

elif(page =="Beta-VAE 2 "):
    
    model = load_model('./decoderbetavae1000_punk.h5',compile = False)
    preds = model.predict(np.random.randn(400,576))

    def plot_images(rows, cols, images):
        grid = np.zeros(shape=(rows*24, cols*24,3))
        for row in range(rows):
            for col in range(cols):
                grid[row*24:(row+1)*24, col*24:(col+1)*24, :] = images[row*cols + col]

        return grid

    st.header(" Beta VAE with individual encoder and decoder")
    n= st.number_input("number of images to be generated",25,400,100)
    number =int( math.sqrt(n))
    img = plot_images(number,number,preds)
    img = np.clip(img,0,1)
    st.image(img,width = 400)
    st.text(str(n)+' Images created using Beta Variational Auto Encoder with latent-space of 576.')
    st.text('In this model I trained the encoder only on KL Loss .')
    st.text('I trained decoder only based on Reconstruction Loss')
    
elif(page == "Latent Space properties"):
    st.header('Latent Space Behaviour of two models')
    col1 , col2,col3,col4 = st.columns([2,2,2,1])
    with col1 :
        st.subheader('Vae Latent_space')
        samples = pkl.load(open('latent_punk.pkl','rb'))
        l = st.number_input('Latent Index',0,575,0,key ='1234')
        sns.distplot(samples[:,l])
        fig=plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(fig)
    with col2 :
        st.subheader('Beta Vae Latent_space')
        samples = pkl.load(open('betalatent_punk.pkl','rb'))
        l2 = st.number_input('Latent Index',0,575,0,key ='123')
        sns.distplot(samples[:,l2])
        fig=plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(fig)
    with col3 :
        st.subheader('Beta Vae own')
        samples = pkl.load(open('betalatent_punk.pkl','rb'))
        l3 = st.number_input('Latent Index',0,575,0,key ='12345')
        sns.distplot(samples[:,l3])
        fig=plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(fig)
        
