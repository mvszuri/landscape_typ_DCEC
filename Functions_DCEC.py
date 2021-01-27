# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:07:06 2020

@author: vmaarten
"""

# This function was adapted from: https://lucehe.github.io/differentiable-argmax/
def prob2oneHot(x):
    import tensorflow as tf
    a = K.pow(x, 10)
    out_sum = tf.reduce_sum(a,axis=3,keepdims=True)
    out_div = tf.divide(a, out_sum)
    return out_div

# This function was adapted from: https://github.com/Tony607/Keras_Deep_Clustering/blob/master/Keras-DEC.ipynb
def autoencoderConv2D(input_dim=(128,128,4), num_layers = 10):
    """
    Conv2D auto-encoder model.
    Arguments:
        img_shape: e.g. (28, 28, 1) for MNIST
        num_layers: number of layers in the hidden layer
    return:
        (autoencoder, encoder), Model of autoencoder and model of encoder
        encoded_shape: the number of units in the hidden layer
    """
    # Convolutional AutoEncoder (Based on U-Net on: https://github.com/zhixuhao/unet/blob/master/model.py)
    # BatchNormalisation with the standard settings should be fine: https://forums.fast.ai/t/batchnormalization-axis-1-when-used-on-convolutional-layers/214/12
    inputs = Input(shape=input_dim)
    #Encoder
    econv = Conv2D(32, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='encoder_0')(inputs)
    #econv = BatchNormalization()(econv)
    econv = MaxPooling2D(pool_size=(2, 2), name='encoder_1')(econv)
    econv = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='encoder_2')(econv)
    #econv = BatchNormalization()(econv)
    econv = MaxPooling2D(pool_size=(2, 2), name='encoder_3')(econv)
    econv = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='encoder_4')(econv)
    #econv = BatchNormalization()(econv)
    econv = MaxPooling2D(pool_size=(2, 2), name='encoder_5')(econv)
    econv = Conv2D(num_layers, 1, activation ='relu', padding = 'same', kernel_initializer = 'he_normal')(econv)
    econv_shape = K.int_shape(econv)[1:4]
    encoded = Flatten(name='encoder_6')(econv)
    encoded_shape = K.int_shape(encoded)[1]
    
    #Decoder
    dconv = Reshape(econv_shape, name='decoder_2')(encoded)
    dconv = UpSampling2D(size = (2,2), name='decoder_3')(dconv)
    dconv = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='decoder_4')(dconv)
    #dconv = BatchNormalization()(dconv)
    dconv = UpSampling2D(size = (2,2), name='decoder_5')(dconv)
    dconv = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='decoder_6')(dconv)
    #dconv = BatchNormalization()(dconv)
    dconv = UpSampling2D(size = (2,2), name='decoder_7')(dconv)
    dconv = Conv2D(32, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='decoder_8')(dconv)
    #dconv = BatchNormalization()(dconv)
    decoded = Conv2D(input_dim[2], 1, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal', name='decoder_9')(dconv)
    
    return Model(inputs=inputs, outputs=decoded, name='AE'), Model(inputs=inputs, outputs=encoded, name='encoder'), encoded_shape

# The below function was adapted from the code belonging to the publication: Guo X, Liu X, Zhu E, Yin J (2017) Deep Clustering with Convolutional Autoencoders, Proceedings of the International Conference on Neural Information Processing, Guangzhou, China.
# Gou et al.'s code can be found here: https://github.com/XifengGuo/DCEC/blob/master/DCEC.py
def CAE(input_dim=(128,128,4), filters=[32, 64, 128, 10]):
    if input_dim[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'

    from keras.layers import Conv2D, Flatten, Dense, Reshape, Conv2DTranspose
    from keras.models import Model
    from keras.engine import Layer, InputSpec, Input
    from keras import backend as K

    inputs = Input(shape=input_dim)
    #Encoder
    econv = Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1')(inputs)
    econv = Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2')(econv)
    econv = Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3')(econv)
    econv_shape = K.int_shape(econv)[1:4]
    flat = Flatten(name='encoder_6')(econv)
    flat_shape = K.int_shape(flat)[1]
    encoded = Dense(units=filters[3], name='embedding')(flat)
    encoded_shape = K.int_shape(encoded)[1]
    
    #Decoder
    dense = Dense(flat_shape, activation='relu')(encoded)
    dconv = Reshape(econv_shape)(dense)
    dconv = Conv2DTranspose(filters[1],3,strides=2,padding=pad3,activation='relu', name='deconv3')(dconv)
    dconv = Conv2DTranspose(filters[0],5,strides=2,padding='same',activation='relu', name='deconv2')(dconv)
    decoded = Conv2DTranspose(input_dim[2],5,strides=2,padding='same',name='deconv1')(dconv)
    
    return Model(inputs=inputs, outputs=decoded, name='AE'), Model(inputs=inputs, outputs=encoded, name='encoder'), encoded_shape

# The below function was adapted from the code belonging to the publication: Guo X, Liu X, Zhu E, Yin J (2017) Deep Clustering with Convolutional Autoencoders, Proceedings of the International Conference on Neural Information Processing, Guangzhou, China.
# Gou et al.'s code can be found here: https://github.com/XifengGuo/DCEC/blob/master/DCEC.py
def CAE2(input_dim=(128,128,4), filters=[32, 64, 128, 256, 10]):
    if input_dim[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'

    from keras.layers import Conv2D, Flatten, Dense, Reshape, Conv2DTranspose
    from keras.models import Model
    from keras.engine import Layer, InputSpec, Input
    from keras import backend as K
    
    inputs = Input(shape=input_dim)
    #Encoder
    econv = Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1')(inputs)
    econv = Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2')(econv)
    econv = Conv2D(filters[2], 5, strides=2, padding='same', activation='relu', name='conv3')(econv)
    econv = Conv2D(filters[3], 3, strides=2, padding=pad3, activation='relu', name='conv4')(econv)
    econv_shape = K.int_shape(econv)[1:4]
    flat = Flatten(name='flatten')(econv)
    flat_shape = K.int_shape(flat)[1]
    encoded = Dense(units=filters[4], name='embedding')(flat)
    encoded_shape = K.int_shape(encoded)[1]
    
    #Decoder
    dense = Dense(flat_shape, activation='relu', name = "Dense")(encoded)
    dconv = Reshape(econv_shape)(dense)
    dconv = Conv2DTranspose(filters[2],3,strides=2,padding=pad3,activation='relu', name='deconv4')(dconv)
    dconv = Conv2DTranspose(filters[1],5,strides=2,padding='same',activation='relu', name='deconv3')(dconv)
    dconv = Conv2DTranspose(filters[0],5,strides=2,padding='same',activation='relu', name='deconv2')(dconv)
    decoded = Conv2DTranspose(input_dim[2],5,strides=2,padding='same',name='deconv1')(dconv)
    
    return Model(inputs=inputs, outputs=decoded, name='AE'), Model(inputs=inputs, outputs=encoded, name='encoder'), encoded_shape

# definition to show original image and reconstructed image
def showOrigDec(orig, dec, savePath, layers, num=10):
    import matplotlib.pyplot as plt
    n = num
    l = layers
    plt.figure(figsize=(20, (l*4)))

    for i in range(n):
        for j in range(l):
            # display original
            ax = plt.subplot((l*2), n, i+1+(n*(j+j)))
            plt.imshow(orig[i][:,:,j], vmin=0, vmax=1, cmap='jet')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
                
            # display reconstruction
            ax = plt.subplot((l*2), n, i+1+(n*(j+j+1)))
            plt.imshow(dec[i][:,:,j], vmin=0, vmax=1, cmap='jet')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
      
    plt.savefig(savePath,dpi = 600)
    plt.show()

# The below functions were adapted from the code belonging to the publication: Guo X, Liu X, Zhu E, Yin J (2017) Deep Clustering with Convolutional Autoencoders, Proceedings of the International Conference on Neural Information Processing, Guangzhou, China.
# Gou et al.'s code can be found here: https://github.com/XifengGuo/DCEC/blob/master/DCEC.py
# computing an auxiliary target distribution
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

def target_distribution2(q):
    weight = q ** 2 #Without normalising for the size of the cluster.
    return (weight.T / weight.sum(1)).T


# Plot a number of random images per detected class
def plotClassExamples(data, labels, savePath, ex_per_class=10):
    import matplotlib.pyplot as plt
    classes = np.unique(labels)
    n_class = classes.shape[0]
    plt.figure(figsize=(ex_per_class*2, n_class*2))
    for i in classes:
        i_choice = np.random.choice(np.where(labels == i)[0], size=ex_per_class, replace=False)
        examp_data = data[i_choice]
        for n in range(ex_per_class):
            # display original
            ax = plt.subplot(n_class, ex_per_class, n+1+(i*ex_per_class))
            plt.imshow(examp_data[n])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        
    plt.savefig(savePath,dpi = 600)
    plt.show()

# Funtion to monitor the progress of the reconstruction.
# Adapted from: https://blender.stackexchange.com/questions/3219/how-to-show-to-the-user-a-progression-in-a-script
def update_progress(job_title, progress):
    import sys
    length = 20 # modify this to change the length
    block = int(round(length*progress))
    msg = "\r{0}: [{1}] {2}%".format(job_title, "#"*block + "-"*(length-block), round(progress*100, 2))
    if progress >= 1: msg += " DONE\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush()
    