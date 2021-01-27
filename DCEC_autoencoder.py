# The code below performs Deep Convolutional Embeded Clustering on image tiles extracted from Switzerland-wide Sentinel rasters and other raster layers.
# Parts of the code were adapted from the code belonging to the publication: Guo X, Liu X, Zhu E, Yin J (2017) Deep Clustering with Convolutional Autoencoders, Proceedings of the International Conference on Neural Information Processing, Guangzhou, China.
# Gou et al.'s code can be found here: https://github.com/XifengGuo/DCEC/blob/master/DCEC.py

# Import packages
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.layers import *
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras.engine import Layer, InputSpec
from keras.optimizers import SGD, Adam
from keras import callbacks
from keras.initializers import VarianceScaling
from keras.utils import plot_model
from IPython.display import Image
from time import time
import pandas as pd
import os, sys, time
import numpy as np
#from keract import get_activations
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, davies_bouldin_score, calinski_harabasz_score, silhouette_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#Settings
saveDir = "C:/PROCESSING/Landscape_typology2/Output10/"  #Output workspace
baseDir = r"D:\Current_work\Projects\Landscape_typologies\SwissWide_analysis" #The folder containing the functions python code
#Input NPZ-file containing all the tiles
inputNPZ = r"D:\Current_work\Projects\Landscape_typologies\SwissWide_data\data_tiles_20200706.npz" 
# Parameters for pre-training
prop_train = 0.85 #Proportion of the data that is used as training data
prop_val = 0.12 #Proportion of the data that is used as validation data. The rest is test data.
epochs = 400
num_layers = 1500 #Number of units in smallest convolution layer
batch_size = 128
#Parameters for DDEV
max_updates = 400
batch_size_DDEV = 256
epoch_per_update = 100
max_epoch = 1800
tol = 0.001 # tolerance threshold to stop training

if not os.path.isdir(saveDir):
    os.makedirs(saveDir)

# Import the functions
os.chdir(baseDir)
from Functions_DDEC_20200622 import *
from ClustLay import *

# Load dataset
with np.load(inputNPZ, allow_pickle=False) as npz_file:
    tiles = dict(npz_file.items())
    
# Convert pixels into floating point numbers
data = tiles['data'].astype(np.float32)
input_dim = data.shape[1:4]

# Standardise the values between 0 and 1
n_bands = data.shape[3]
for band in np.arange(n_bands):
    max_val = np.max(data[:,:,:,band])
    data[:,:,:,band] = data[:,:,:,band]/max_val
   
# Make training, validation and test datasets
size_train = np.int(data.shape[0]*prop_train)
size_val = np.int(data.shape[0]*prop_val)
x_train,x_val,x_test = np.vsplit(data[np.random.permutation(data.shape[0])],(size_train,(size_train+size_val)))

#Pre-train AutoEncoder
autoencoder, encoder, num_units = CAE(input_dim=input_dim, filters=[32, 64, 128, num_layers])
 #autoencoder, encoder, num_units = CAE2(input_dim=input_dim, filters=[32, 64, 128, 256, num_layers])
autoencoder.summary()

adam = Adam(amsgrad=True)
autoencoder.compile(adam, loss='mse') # optimizer='adam'

# load pretrained weights
# model.load_weights(saveDir + "AutoEncoder_LandsTopo_Deep_weights.01-0.62-0.61.hdf5")

es_cb = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
rop_cb = ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=9,verbose=1,mode="auto",min_lr=0.00001)
chkpt = saveDir + 'AutoEncoder_LandsTopo_Deep_weights.{epoch:02d}-{loss:.4f}-{val_loss:.4f}.hdf5'
cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = autoencoder.fit(x_train, x_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_val, x_val),
                    callbacks=[es_cb,rop_cb],
                    shuffle=True)

# Save the weights and the training history
autoencoder.save_weights(saveDir+'AutoEncoder_LandsTopo_l'+str(num_layers)+'_final_model.h5')
pd.DataFrame(history.history).to_csv(saveDir+'AutoEncoder_LandsTopo_l'+str(num_layers)+'_trainHist.csv', index=False)

# Evaluate with test dataset
score = autoencoder.evaluate(x_test, x_test, verbose=1)
print(score)

# Visualize original image and reconstructed image
x_test_sel = x_test[0:10]
x_val_sel = x_val[0:10]
x_test_pred = autoencoder.predict(x_test_sel)
x_val_pred = autoencoder.predict(x_val_sel)   
showOrigDec(x_test_sel, x_test_pred, savePath = saveDir+'orig_reconstucted_test_l'+str(num_layers)+'.png', layers = input_dim[2])
showOrigDec(x_val_sel, x_val_pred, savePath = saveDir+'orig_reconstucted_val_l'+str(num_layers)+'.png', layers = input_dim[2])

# Set the arrays in which the training evaluation indicators will be stored
run_arr = np.array([])
Nclust_arr = np.array([])
time_arr = np.array([])
DB_arr = np.array([])
S_arr = np.array([])
CH_arr = np.array([])
final_loss_arr = np.array([])
final_clust_loss_arr = np.array([])
final_dec_loss_arr = np.array([])
Nepoch_arr = np.array([])
LR_arr = np.array([])

# Loop over the repeats (training is repeated for each hyperparameter setting 4 times).
for run in [4]:
    # Loop over the potential numbers of clusters
    for n_clusters in [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]:
        # Start timer for run
        start_time = time.time()
        
        # load pretrained weights
        autoencoder.load_weights(saveDir+'AutoEncoder_LandsTopo_l'+str(num_layers)+'_final_model.h5')
        clustering_layer = ClusteringLayer(n_clusters, n_units=num_units, name='clustering')(encoder.output)
        model = Model(inputs=encoder.input,outputs=[clustering_layer, autoencoder.output])
        plot_model(model, to_file=saveDir+'model.png', show_shapes=True)
        Image(filename=saveDir+'model.png')
        
        # Initialize cluster centers using k-means
        #kmeans = MiniBatchKMeans(n_clusters=n_clusters,verbose = 1,batch_size=20*batch_size)
        kmeans = KMeans(n_clusters=n_clusters, verbose = 1) #n_init=20
        print('Estimating K-means')
        y_pred_kmeans = kmeans.fit_predict(encoder.predict(data, verbose=1))
        model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
        y_pred_last = np.copy(y_pred_kmeans)
        
        #Set the optimiser
        learn_rate=0.0005
        adam = Adam(amsgrad=True, learning_rate=learn_rate)
        model.compile(adam, loss=['kld', 'mse'], loss_weights=[0.1, 1])
        
        #model.load_weights(saveDir+'conv_b_DEC_model_update0.hdf5')
        
        # Set the epoch counter to 0. This is used for continuous epochs counting after updates. Adapted from: https://stackoverflow.com/questions/49409448/how-training-rate-changes-between-epochs-in-keras-tensorflow
        epoch_counter = 0
        # The clust_loss_counter records the number of times the clustering loss is not improved.
        clust_loss_counter = 0
        prev_min_clust_loss = 1
        # Set the arrays in which the training evaluation indicators will be stored
        update_arr = np.array([])
        epoch_arr = np.array([])
        loss_arr = np.array([])
        clust_loss_arr = np.array([])
        dec_loss_arr = np.array([])
        learn_rate_arr = np.array([])
        deltaLab_arr = np.array([])
        
        #Set the callbacks
        es_ddec = EarlyStopping(monitor='loss', patience=5, verbose=1, mode='auto', restore_best_weights=True)
        # rop_ddec = ReduceLROnPlateau(monitor ='clustering_loss', factor = 0.1, patience=8, verbose = 1, mode = 'auto', min_lr=1.0e-08)
        
        for nr_update in range(max_updates):
            #model.save_weights(saveDir+'conv_b_DEC_model_update'+str(nr_update)+'.hdf5')
        
            print('################################ UPDATE:', nr_update)
            print('Predicting labels')
            q, _  = model.predict(data, verbose=1)
            p = target_distribution2(q)  # update the auxiliary target distribution p
        
            # evaluate the clustering performance
            y_pred = q.argmax(1)
            
            # Calculate the Delta-label (i.e. difference in allocations between two consecutive updates)
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = np.copy(y_pred)
            print('Delta lable: ', delta_label)
            
            #Plot the first two principle components and the classes.
            if nr_update % 3 == 0:
                print("######### Producing intermediate PCA-plot")
                x_hidden = encoder.predict(data)
                x_hidden_std = StandardScaler().fit_transform(x_hidden)
                pca=PCA(2)
                X_pca = pca.fit_transform(x_hidden_std)
                var_expl = np.sum(pca.explained_variance_ratio_)
                DB_score = davies_bouldin_score(x_hidden_std, y_pred)
                plt.scatter(X_pca[:,0], X_pca[:,1], c=y_pred, cmap = "jet")
                plt.colorbar()
                plt.title("Expl. var.: "+str(np.round(var_expl,5))+". DB score: "+str(np.round(DB_score, 5)))
                plt.savefig(saveDir+'PCA_hidden_data_yDEC_r'+str(run)+'_c'+str(n_clusters)+'_e'+str(epoch_counter)+'.png', dpi=300)
                plt.show()
            
            # Reduce the learning rate if there is no improvement in the clustering loss
            if clust_loss_counter == 3:
                learn_rate = learn_rate*0.5
                K.set_value(model.optimizer.learning_rate, learn_rate)
                print("!!!!!Learning rate reduced to:", K.eval(model.optimizer.lr))
                clust_loss_counter = 0
            
            # Calculate the maximum number of epochs of this update
            epoch_max = epoch_counter + epoch_per_update
            loss = model.fit(x=data,
                             y=[p,data],
                             batch_size=batch_size,
                             initial_epoch=epoch_counter,
                             epochs=epoch_max,
                             verbose=1,
                             callbacks=[es_ddec],
                             shuffle=True)
          
            #Update the epoch counter
            epoch_counter = epoch_counter + len(loss.epoch)
            
            #Check if the clustering loss has imporved and update clust_loss_counter
            min_clust_loss = min(loss.history['loss'])
            if min_clust_loss >= prev_min_clust_loss:
                clust_loss_counter = clust_loss_counter + 1
                print("Loss not improved")
            else:
                clust_loss_counter = 0
            print("Clustering loss counter:", clust_loss_counter)
            prev_min_clust_loss = min_clust_loss
            
            #Write statistics to log
            update_arr = np.append(update_arr, np.repeat(nr_update, len(loss.epoch)))
            epoch_arr = np.append(epoch_arr, loss.epoch)
            loss_arr = np.append(loss_arr, loss.history['loss'])
            clust_loss_arr = np.append(clust_loss_arr, loss.history['clustering_loss'])
            dec_loss_arr = np.append(dec_loss_arr, loss.history['deconv1_loss'])
            learn_rate_arr = np.append(learn_rate_arr, np.repeat(K.eval(model.optimizer.lr), len(loss.epoch)))
            deltaLab_arr = np.append(deltaLab_arr, np.repeat(delta_label, len(loss.epoch)))
            
            if epoch_counter > max_epoch:
                print('Reached maximum number of epochs. Stopping training.')
                break
        
        # Save the weights and the training history
        model.save_weights(saveDir+'DDEC_model_final_r'+str(run)+'_c'+str(n_clusters)+'.h5')
        results_df = pd.DataFrame({"Update":update_arr,
                      "Epoch":epoch_arr,
                      "Loss":loss_arr,
                      "Clust_loss":clust_loss_arr,
                      "Dec_loss":dec_loss_arr,
                      "Learning_rate":learn_rate_arr,
                      "Delta_lable":deltaLab_arr})
        results_df.to_csv(saveDir+'DDEC_training_history_r'+str(run)+'_c'+str(n_clusters)+'.csv', index=False)
        
        #model.load_weights(saveDir+'DDEC_model_final_c'+str(n_clusters)+'.h5')
        
        # Plot the model clusters on the first two components of a PCA
        # Adapted from: https://shankarmsy.github.io/posts/pca-sklearn.html
        x_train_std = np.reshape(data, (data.shape[0], data.shape[1]*data.shape[2]*data.shape[3]))
        x_train_std = StandardScaler().fit_transform(x_train_std)
        
        # fit on the whole data set
        q, _  = model.predict(data, verbose=1)
        y_pred_DEC = q.argmax(1)
        
        pca=PCA(2)
        X_pca = pca.fit_transform(x_train_std)
        var_expl = np.sum(pca.explained_variance_ratio_)
        DB_score = davies_bouldin_score(x_train_std, y_pred_kmeans)
        plt.scatter(X_pca[:,0], X_pca[:,1], c=y_pred_kmeans, cmap = "jet")
        plt.colorbar()
        plt.title("Expl. var.: "+str(np.round(var_expl,5))+". DB score: "+str(np.round(DB_score, 5)))
        plt.savefig(saveDir+'PCA_Orig_data_kmeans_r'+str(run)+'_c'+str(n_clusters)+'.png',dpi=300)
        plt.show()
        
        pca=PCA(2)
        DB_score = davies_bouldin_score(x_train_std, y_pred_DEC)
        plt.scatter(X_pca[:,0], X_pca[:,1], c=y_pred_DEC, cmap = "jet")
        plt.colorbar()
        plt.title("Expl. var.: "+str(np.round(var_expl,5))+". DB score: "+str(np.round(DB_score, 5)))
        plt.savefig(saveDir+'PCA_Orig_data_yDEC_r'+str(run)+'_c'+str(n_clusters)+'.png', dpi=300)
        plt.show()
        
        x_hidden = encoder.predict(data, verbose = 1)
        x_hidden_std = StandardScaler().fit_transform(x_hidden)
        pca=PCA(2)
        X_pca = pca.fit_transform(x_hidden_std)
        var_expl = np.sum(pca.explained_variance_ratio_)
        DB_score = davies_bouldin_score(x_hidden_std, y_pred_DEC)
        plt.scatter(X_pca[:,0], X_pca[:,1], c=y_pred_DEC, cmap = "jet")
        plt.colorbar()
        plt.title("Expl. var.: "+str(np.round(var_expl,5))+". DB score: "+str(np.round(DB_score, 5)))
        plt.savefig(saveDir+'PCA_hidden_data_yDEC_r'+str(run)+'_c'+str(n_clusters)+'.png', dpi=300)
        plt.show()
        
        #Save the true and reconstructed images after the autoencoder has been updated
        x_test_sel = x_test[0:10]
        x_test_pred = autoencoder.predict(x_test_sel)
        showOrigDec(x_test_sel, x_test_pred, savePath = saveDir+'orig_reconstucted_DEC_test_r'+str(run)+'_c'+str(n_clusters)+'.png', layers = input_dim[2])
        
        # Save the model predictions to a text file
        np.savetxt(saveDir+'DDEC_preds_r'+str(run)+'_c'+str(n_clusters)+'.csv', y_pred_DEC, delimiter=",")
        
        # Calculate further scores of clustering success
        S_score = silhouette_score(x_hidden_std, y_pred_DEC)
        CH_score = calinski_harabasz_score(x_hidden_std, y_pred_DEC)
        
        # Stop the time calculation
        run_time = time.time() - start_time
        
        # Write run scores to vectors
        run_arr = np.append(run_arr, run)
        Nclust_arr = np.append(Nclust_arr, n_clusters)
        time_arr = np.append(time_arr, run_time)
        DB_arr = np.append(DB_arr, DB_score)
        S_arr = np.append(S_arr, S_score)
        CH_arr = np.append(CH_arr, CH_score)
        final_loss_arr = np.append(final_loss_arr, np.min(loss.history['loss']))
        final_clust_loss_arr = np.append(final_clust_loss_arr, np.min(loss.history['clustering_loss']))
        final_dec_loss_arr = np.append(final_dec_loss_arr, np.min(loss.history['deconv1_loss']))
        Nepoch_arr = np.append(Nepoch_arr, np.max(loss.epoch))
        LR_arr = np.append(LR_arr, K.eval(model.optimizer.lr))
        
        final_results_df = pd.DataFrame({"Run":run_arr,
                      "N_Clust":Nclust_arr,
                      "Run_time":time_arr,
                      "DB_score":DB_arr,
                      "S_score":S_arr,
                      "CH_score":CH_arr,
                      "Loss":final_loss_arr,
                      "Clust_loss":final_clust_loss_arr,
                      "Dec_loss":final_dec_loss_arr,
                      "N_epoch":Nepoch_arr,
                      "Learning_rate":LR_arr})
        final_results_df.to_csv(saveDir+'Final_results.csv', index=False)


