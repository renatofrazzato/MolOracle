import pickle
import numpy as np
#import scipy as sc
import time
#import pandas as pd
import os
import gc
import math as m
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from BatchDataFunctions import GerarBatch
from BatchDataFunctions import GetFeaturesPaddingAdjustments
from BatchDataFunctions import GetDistPaddingAdjustments



##--------------- Auxiliary Functions ------------------

## activation function --> shifted soft plus
def ssp(param_valor):
  return tf.math.softplus(param_valor) - tf.math.log(tf.cast(2, tf.dtypes.float64))

## Decay function. This function decreases when distance increases
def cutoff_distance(param_batch_dist, param_weight_dist_cutoff):
  val_cutoff = param_weight_dist_cutoff[0,0]
  final_result_0 = 1 - 6*((param_batch_dist/val_cutoff)**5) + 15*((param_batch_dist/val_cutoff)**4) - 10*((param_batch_dist/val_cutoff)**3)
  final_result = tf.nn.relu(final_result_0)

  return final_result

## Expand distances using radial basis function
def DistanceExpansion(param_batch_dist,param_Weights1_rbf_mu, param_Weights1_rbf_gamma):
  aux_dist_expand = tf.expand_dims(param_batch_dist, 3)
  ## RBF expansion calculation
  aux_dist_expand0 = (aux_dist_expand - param_Weights1_rbf_mu)**2
  rbf_expansion = tf.math.exp(-1.0*tf.multiply(aux_dist_expand0, param_Weights1_rbf_gamma))

  return rbf_expansion


## ----------------------------------------------------------

max_num_atm = 29 ## QM9

initial_feature_dim = 5 ## QM9

dist_expand_dim = 300

neurons_quantity = 128


############### Placeholders: Used for input data into the model pipeline ###############
tf.compat.v1.reset_default_graph() ### make sure the graph is cleaned

## Placeholders ----> used for input the data in the model
plc_batch_features = tf.placeholder(dtype=tf.float64, shape=(None, max_num_atm, initial_feature_dim))
plc_batch_dist = tf.placeholder(dtype=tf.float64, shape=(None, max_num_atm, max_num_atm))
plc_batch_target = tf.placeholder(dtype=tf.float64, shape=(None, 1))

# The molecules are represented as matrices, to be able to create the batches all the matrices need to have the same dimensions so, 
# padding is applied in order to standardize the molecule sizes.
plc_batch_bias_dist = tf.placeholder(dtype=tf.float64, shape=(None, max_num_atm, max_num_atm, 1))  ### This is used for padding
plc_batch_bias = tf.placeholder(dtype=tf.float64, shape=(None, max_num_atm, 1)) ### This is used for padding
#######################################################################################

## ------------------->>>>Parameters for disance: RBF and Cutoff<<<<-------------------------

Weights1_rbf_mu = tf.constant(np.array([[(x)*(1/10) for x in range(dist_expand_dim)]]), dtype=tf.dtypes.float64)

Weights1_rbf_gamma = tf.constant(np.array([[10.0]]), dtype=tf.dtypes.float64)

Weight_dist_cutoff = tf.Variable(tf.random.normal(shape=(1,1), mean=7.0, stddev=1, dtype=tf.dtypes.float64), name='Weight_dist_cutoff')

## ------------------->>>> Parameters for Embeddings <<<<-------------------------
## Each Chemical element is represented by an embedding.
# Embeddings are randomly initialized based on atomic numbers.
Weights_embeddings = tf.Variable(tf.random.normal(shape=(initial_feature_dim, neurons_quantity), mean=[[6],[9],[1],[7],[8]], stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='Weights_embeddings')


########################### Parameters for interaction Modules ###############################

## ------------------->>>> First Interaction <<<<-------------------------
Weights1_NMP_01_distance = tf.Variable(tf.random.normal(shape=(dist_expand_dim, neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='Weights1_NMP_01_distance')
bias1_NMP_01_distance = tf.Variable(tf.random.normal(shape=(1,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='bias1_NMP_01_distance')

Weights2_NMP_01_distance = tf.Variable(tf.random.normal(shape=(neurons_quantity,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='Weights2_NMP_01_distance')
bias2_NMP_01_distance = tf.Variable(tf.random.normal(shape=(1,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='bias2_NMP_01_distance')


Weights1_NMP_01_aggregation = tf.Variable(tf.random.normal(shape=(neurons_quantity,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='Weights1_NMP_01_aggregation')
bias1_NMP_01_aggregation = tf.Variable(tf.random.normal(shape=(1,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='bias1_NMP_01_aggregation')

Weights2_NMP_01_aggregation = tf.Variable(tf.random.normal(shape=(neurons_quantity,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='Weights2_NMP_01_aggregation')
bias2_NMP_01_aggregation = tf.Variable(tf.random.normal(shape=(1,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='bias2_NMP_01_aggregation')


## ------------------->>>> Second Interaction <<<<-------------------------
Weights1_NMP_02_distance = tf.Variable(tf.random.normal(shape=(dist_expand_dim,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='Weights1_NMP_02_distance')
bias1_NMP_02_distance = tf.Variable(tf.random.normal(shape=(1,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='bias1_NMP_02_distance')

Weights2_NMP_02_distance = tf.Variable(tf.random.normal(shape=(neurons_quantity,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='Weights2_NMP_02_distance')
bias2_NMP_02_distance = tf.Variable(tf.random.normal(shape=(1,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='bias2_NMP_02_distance')


Weights1_NMP_02_aggregation = tf.Variable(tf.random.normal(shape=(neurons_quantity,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='Weights1_NMP_02_aggregation')
bias1_NMP_02_aggregation = tf.Variable(tf.random.normal(shape=(1,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='bias1_NMP_02_aggregation')

Weights2_NMP_02_aggregation = tf.Variable(tf.random.normal(shape=(neurons_quantity,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='Weights2_NMP_02_aggregation')
bias2_NMP_02_aggregation = tf.Variable(tf.random.normal(shape=(1,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='bias2_NMP_02_aggregation')

## ------------------->>>> Third Interaction <<<<-------------------------
Weights1_NMP_03_distance = tf.Variable(tf.random.normal(shape=(dist_expand_dim,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='Weights1_NMP_03_distance')
bias1_NMP_03_distance = tf.Variable(tf.random.normal(shape=(1,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='bias1_NMP_03_distance')

Weights2_NMP_03_distance = tf.Variable(tf.random.normal(shape=(neurons_quantity,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='Weights2_NMP_03_distance')
bias2_NMP_03_distance = tf.Variable(tf.random.normal(shape=(1,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='bias2_NMP_03_distance')


Weights1_NMP_03_aggregation = tf.Variable(tf.random.normal(shape=(neurons_quantity,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='Weights1_NMP_03_aggregation')
bias1_NMP_03_aggregation = tf.Variable(tf.random.normal(shape=(1,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='bias1_NMP_03_aggregation')

Weights2_NMP_03_aggregation = tf.Variable(tf.random.normal(shape=(neurons_quantity,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='Weights2_NMP_03_aggregation')
bias2_NMP_03_aggregation = tf.Variable(tf.random.normal(shape=(1,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='bias2_NMP_03_aggregation')

## ------------------->>>> Fourth Interaction <<<<-------------------------
Weights1_NMP_04_distance = tf.Variable(tf.random.normal(shape=(dist_expand_dim,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='Weights1_NMP_04_distance')
bias1_NMP_04_distance = tf.Variable(tf.random.normal(shape=(1,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='bias1_NMP_04_distance')

Weights2_NMP_04_distance = tf.Variable(tf.random.normal(shape=(neurons_quantity,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='Weights2_NMP_04_distance')
bias2_NMP_04_distance = tf.Variable(tf.random.normal(shape=(1,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='bias2_NMP_04_distance')


Weights1_NMP_04_aggregation = tf.Variable(tf.random.normal(shape=(neurons_quantity,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='Weights1_NMP_04_aggregation')
bias1_NMP_04_aggregation = tf.Variable(tf.random.normal(shape=(1,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='bias1_NMP_04_aggregation')

Weights2_NMP_04_aggregation = tf.Variable(tf.random.normal(shape=(neurons_quantity,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='Weights2_NMP_04_aggregation')
bias2_NMP_04_aggregation = tf.Variable(tf.random.normal(shape=(1,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='bias2_NMP_04_aggregation')

## ------------------->>>> Fith Interaction <<<<-------------------------
Weights1_NMP_05_distance = tf.Variable(tf.random.normal(shape=(dist_expand_dim,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='Weights1_NMP_05_distance')
bias1_NMP_05_distance = tf.Variable(tf.random.normal(shape=(1,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='bias1_NMP_05_distance')

Weights2_NMP_05_distance = tf.Variable(tf.random.normal(shape=(neurons_quantity,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='Weights2_NMP_05_distance')
bias2_NMP_05_distance = tf.Variable(tf.random.normal(shape=(1,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='bias2_NMP_05_distance')


Weights1_NMP_05_aggregation = tf.Variable(tf.random.normal(shape=(neurons_quantity,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='Weights1_NMP_05_aggregation')
bias1_NMP_05_aggregation = tf.Variable(tf.random.normal(shape=(1,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='bias1_NMP_05_aggregation')

Weights2_NMP_05_aggregation = tf.Variable(tf.random.normal(shape=(neurons_quantity,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='Weights2_NMP_05_aggregation')
bias2_NMP_05_aggregation = tf.Variable(tf.random.normal(shape=(1,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='bias2_NMP_05_aggregation')

## ------------------->>>> Sixth Interaction <<<<-------------------------
Weights1_NMP_06_distance = tf.Variable(tf.random.normal(shape=(dist_expand_dim,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='Weights1_NMP_06_distance')
bias1_NMP_06_distance = tf.Variable(tf.random.normal(shape=(1,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='bias1_NMP_06_distance')

Weights2_NMP_06_distance = tf.Variable(tf.random.normal(shape=(neurons_quantity,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='Weights2_NMP_06_distance')
bias2_NMP_06_distance = tf.Variable(tf.random.normal(shape=(1,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='bias2_NMP_06_distance')


Weights1_NMP_06_aggregation = tf.Variable(tf.random.normal(shape=(neurons_quantity,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='Weights1_NMP_06_aggregation')
bias1_NMP_06_aggregation = tf.Variable(tf.random.normal(shape=(1,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='bias1_NMP_06_aggregation')

Weights2_NMP_06_aggregation = tf.Variable(tf.random.normal(shape=(neurons_quantity,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='Weights2_NMP_06_aggregation')
bias2_NMP_06_aggregation = tf.Variable(tf.random.normal(shape=(1,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='bias2_NMP_06_aggregation')

###################   Parameters for atomwise layers #####################

## ------------------->>>> First Dense Layer <<<<-------------------------
Weights1_Dense_01 = tf.Variable(tf.random.normal(shape=(neurons_quantity,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='Weights1_Dense_01')
bias1_Dense_01 = tf.Variable(tf.random.normal(shape=(1,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='bias1_Dense_01')

Weights2_Dense_01 = tf.Variable(tf.random.normal(shape=(neurons_quantity,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='Weights2_Dense_01')
bias2_Dense_01 = tf.Variable(tf.random.normal(shape=(1,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='bias2_Dense_01')

## ------------------->>>> Second Dense Layer <<<<-------------------------
Weights1_Dense_02 = tf.Variable(tf.random.normal(shape=(neurons_quantity,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='Weights1_Dense_02')
bias1_Dense_02 = tf.Variable(tf.random.normal(shape=(1,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='bias1_Dense_02')

Weights2_Dense_02 = tf.Variable(tf.random.normal(shape=(neurons_quantity,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='Weights2_Dense_02')
bias2_Dense_02 = tf.Variable(tf.random.normal(shape=(1,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='bias2_Dense_02')

##########################################################################

## ------------------->>>> Gate Layers <<<<-------------------------
#Weights_Gate_Embeddings = tf.Variable(tf.random.normal(shape=(1,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='Weights_Gate_Embeddings')
Weights_Gate_Layer_01 = tf.Variable(tf.random.normal(shape=(1,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='Weights_Gate_Layer_01')
Weights_Gate_Layer_02 = tf.Variable(tf.random.normal(shape=(1,neurons_quantity), mean=0.0, stddev=1/np.sqrt(neurons_quantity), dtype=tf.dtypes.float64), name='Weights_Gate_Layer_02')

## ------------------->>>> Prediction Layer <<<<-------------------------
Weights1_Prediction = tf.Variable(tf.random.normal(shape=(neurons_quantity,32), mean=0.0, stddev=1/np.sqrt(32), dtype=tf.dtypes.float64), name='Weights1_Prediction')
bias1_prediction = tf.Variable(tf.random.normal(shape=(1,32), mean=0.0, stddev=1/np.sqrt(32), dtype=tf.dtypes.float64), name='bias1_prediction')

Weights2_Prediction = tf.Variable(tf.random.normal(shape=(32,1), mean=0.0, stddev=1/np.sqrt(32), dtype=tf.dtypes.float64), name='Weights2_Prediction')
bias2_Prediction = tf.Variable(tf.random.normal(shape=(1,1), mean=0.0, stddev=1/np.sqrt(32), dtype=tf.dtypes.float64), name='bias2_Prediction')

### ---------------------->>> Function to implement interaction module <<<----------------------------------------------

def NMP(param_batch_features, param_batch_dist, param_batch_distance_expand, param_plc_bias_dist, param_plc_bias,
        param_dist_weights_1, param_dist_weights_1_bias,
        param_dist_weights_2, param_dist_weights_2_bias,
        param_agg_weights_1, param_agg_weights_1_bias,
        param_agg_weights_2, param_agg_weights_2_bias,
        param_weight_dist_cutoff):

    dist_step1 = ssp(tf.matmul(param_batch_distance_expand, param_dist_weights_1) + param_dist_weights_1_bias)
    dist_step2 = tf.matmul(dist_step1, param_dist_weights_2) + param_dist_weights_2_bias

    feat_dist_interaction = tf.multiply(dist_step2, tf.expand_dims(param_batch_features, axis=1))


    batch_dist_cutoff = cutoff_distance(param_batch_dist, param_weight_dist_cutoff)
    aux_dist_expand = tf.expand_dims(batch_dist_cutoff, 3)
    feat_dist_interaction_radius_cutoff = tf.multiply(feat_dist_interaction, aux_dist_expand)


    ## Removing self atoms interactions and removing padding atoms.
    feat_dist_interaction_radius_cutoff_2 = tf.multiply(feat_dist_interaction_radius_cutoff, param_plc_bias_dist)

    #Aggregating the atoms neighbors information
    v = tf.reduce_sum(feat_dist_interaction_radius_cutoff_2, axis=2)

    message_1 = ssp(tf.matmul(v, param_agg_weights_1) + param_agg_weights_1_bias)
    message_2 = tf.matmul(message_1, param_agg_weights_2) + param_agg_weights_2_bias
    message_3 = tf.multiply(message_2, param_plc_bias) ## removing the bias for the padding atoms.

    retorno = param_batch_features + message_3

    return retorno

def AtomNeuralNet():
    #### ------------------------------ >>> Neural Network Model <<< ------------------------------------------
    embeddings = tf.matmul(plc_batch_features, Weights_embeddings)

    rbf_distance_expand = DistanceExpansion(plc_batch_dist, Weights1_rbf_mu, Weights1_rbf_gamma)

    interaction_1 = NMP(embeddings, plc_batch_dist, rbf_distance_expand, plc_batch_bias_dist, plc_batch_bias,
                        Weights1_NMP_01_distance, bias1_NMP_01_distance,
                        Weights2_NMP_01_distance, bias2_NMP_01_distance,
                        Weights1_NMP_01_aggregation, bias1_NMP_01_aggregation,
                        Weights2_NMP_01_aggregation, bias2_NMP_01_aggregation,
                        Weight_dist_cutoff)

    interaction_2 = NMP(interaction_1, plc_batch_dist, rbf_distance_expand, plc_batch_bias_dist, plc_batch_bias,
                        Weights1_NMP_02_distance, bias1_NMP_02_distance,
                        Weights2_NMP_02_distance, bias2_NMP_02_distance,
                        Weights1_NMP_02_aggregation, bias1_NMP_02_aggregation,
                        Weights2_NMP_02_aggregation, bias2_NMP_02_aggregation,
                        Weight_dist_cutoff)

    interaction_3 = NMP(interaction_2, plc_batch_dist, rbf_distance_expand, plc_batch_bias_dist, plc_batch_bias,
                        Weights1_NMP_03_distance, bias1_NMP_03_distance,
                        Weights2_NMP_03_distance, bias2_NMP_03_distance,
                        Weights1_NMP_03_aggregation, bias1_NMP_03_aggregation,
                        Weights2_NMP_03_aggregation, bias2_NMP_03_aggregation,
                        Weight_dist_cutoff)

    dense_layer1_step_0 = ssp(tf.matmul(interaction_3, Weights1_Dense_01) + bias1_Dense_01)
    dense_layer1_step_1 = tf.matmul(dense_layer1_step_0, Weights2_Dense_01) + bias2_Dense_01
    dense_layer1 = tf.multiply(dense_layer1_step_1, plc_batch_bias)

    interaction_4 = NMP(dense_layer1, plc_batch_dist, rbf_distance_expand, plc_batch_bias_dist, plc_batch_bias,
                        Weights1_NMP_04_distance, bias1_NMP_04_distance,
                        Weights2_NMP_04_distance, bias2_NMP_04_distance,
                        Weights1_NMP_04_aggregation, bias1_NMP_04_aggregation,
                        Weights2_NMP_04_aggregation, bias2_NMP_04_aggregation,
                        Weight_dist_cutoff)

    interaction_5 = NMP(interaction_4, plc_batch_dist, rbf_distance_expand, plc_batch_bias_dist, plc_batch_bias,
                        Weights1_NMP_05_distance, bias1_NMP_05_distance,
                        Weights2_NMP_05_distance, bias2_NMP_05_distance,
                        Weights1_NMP_05_aggregation, bias1_NMP_05_aggregation,
                        Weights2_NMP_05_aggregation, bias2_NMP_05_aggregation,
                        Weight_dist_cutoff)

    interaction_6 = NMP(interaction_5, plc_batch_dist, rbf_distance_expand, plc_batch_bias_dist, plc_batch_bias,
                        Weights1_NMP_06_distance, bias1_NMP_06_distance,
                        Weights2_NMP_06_distance, bias2_NMP_06_distance,
                        Weights1_NMP_06_aggregation, bias1_NMP_06_aggregation,
                        Weights2_NMP_06_aggregation, bias2_NMP_06_aggregation,
                        Weight_dist_cutoff)

    dense_layer2_step_0 = ssp(tf.matmul(interaction_6, Weights1_Dense_02) + bias1_Dense_02)
    dense_layer2_step_1 = tf.matmul(dense_layer2_step_0, Weights2_Dense_02) + bias2_Dense_02
    dense_layer2 = tf.multiply(dense_layer2_step_1, plc_batch_bias)

    #gate_layer = tf.multiply(embeddings, Weights_Gate_Embeddings) + tf.multiply(dense_layer1, Weights_Gate_Layer_01) + tf.multiply(dense_layer2, Weights_Gate_Layer_02)
    gate_layer = tf.multiply(dense_layer1, Weights_Gate_Layer_01) + tf.multiply(dense_layer2, Weights_Gate_Layer_02)

    ## ------------->> Prediction <<----------------
    prediction_0 = ssp(tf.matmul(gate_layer, Weights1_Prediction) + bias1_prediction)
    prediction_1 = tf.matmul(prediction_0, Weights2_Prediction) + bias2_Prediction
    prediction = tf.multiply(prediction_1, plc_batch_bias)

    ### ---->>> Final Aggregation <<<---------------
    #output = tf.divide(tf.reduce_sum(prediction, axis=1), tf.reduce_sum(plc_batch_bias, axis=1)) #mean
    output = tf.reduce_sum(prediction, axis=1)

    return output


def TrainModel(param_model, lst_features_treino, 
               lst_target_treino, lst_distancias_treino, 
               lst_mol_sizes_treino, std_dev_atom,
               lst_features_valid, lst_target_valid,
               lst_distancias_valid, lst_mol_sizes_valid, 
               n_epochs, n_batch):
    
    # In some cases the model is trained using transformed target then, 
    # lst_ref_atms_treino, mean_value_atom and std_dev_atom are applied to get the original target values

    erro_eqm = tf.losses.mean_squared_error(plc_batch_target, param_model)
    obj_opt = tf.train.AdamOptimizer(learning_rate=0.0001)
    treinar = obj_opt.minimize(erro_eqm)

    #n_batch = 64
    #n_epochs = 600
    n_batch_per_epoch = int(len(lst_target_treino)/n_batch)

    if not os.path.exists('parameters'):
        os.makedirs('./parameters')

    salvarParametros = tf.train.Saver()
    caminho_parametros = "./parameters/parameters.ckp"


    init = tf.global_variables_initializer()
    with tf.Session() as tf_sess:
        tf_sess.run(init)
        for i in range(n_epochs):
          inicio = time.time()
          
          for r in range(n_batch_per_epoch):
            batch_features, batch_dist, batch_target, batch_features_padding, batch_cfconv_padding = GerarBatch(lst_features_treino, lst_target_treino, 
                                                                                                                lst_distancias_treino, lst_mol_sizes_treino, 
                                                                                                                n_batch, max_num_atm)
            
            tf_sess.run(treinar, feed_dict = {plc_batch_features: batch_features,
                                              plc_batch_dist: batch_dist,
                                              plc_batch_bias: batch_features_padding,
                                              plc_batch_bias_dist: batch_cfconv_padding,
                                              plc_batch_target:batch_target})
            
          batch_features, batch_dist, batch_target, batch_features_padding, batch_cfconv_padding = GerarBatch(lst_features_treino, lst_target_treino, 
                                                                                                                    lst_distancias_treino, lst_mol_sizes_treino, 
                                                                                                                    n_batch, max_num_atm)
          pdct = tf_sess.run(param_model, feed_dict = {plc_batch_features: batch_features,
                                                             plc_batch_dist: batch_dist,
                                                             plc_batch_bias: batch_features_padding,
                                                             plc_batch_bias_dist: batch_cfconv_padding})
                
          mae_train = np.mean(abs((pdct.flatten() - batch_target.flatten())*std_dev_atom))



          batch_features, batch_dist, batch_target, batch_features_padding, batch_cfconv_padding = GerarBatch(lst_features_valid, lst_target_valid, 
                                                                                                                    lst_distancias_valid, lst_mol_sizes_valid, 
                                                                                                                    n_batch, max_num_atm)
          pdct = tf_sess.run(param_model, feed_dict = {plc_batch_features: batch_features,
                                                             plc_batch_dist: batch_dist,
                                                             plc_batch_bias: batch_features_padding,
                                                             plc_batch_bias_dist: batch_cfconv_padding})
          mae_valid = np.mean(abs((pdct.flatten() - batch_target.flatten())*std_dev_atom))
                
                
          std_val = np.std(abs((pdct.flatten() - batch_target.flatten())*std_dev_atom))
          ub = mae_valid + 1.0*std_val/np.sqrt(len(batch_target.flatten()))
                
          params_updated = ''
          if i ==0:
            salvarParametros.save(tf_sess, caminho_parametros)
            mae_anterior = mae_valid
          else:
            if ub < mae_anterior:
              salvarParametros.save(tf_sess, caminho_parametros)
              mae_anterior = mae_valid
              params_updated = '  Weights Updated :-)'

          fim = time.time()
                
          print("Epoch:", i+1, "MAE Train:", mae_train, "      MAE Validation:", mae_valid, "      Time(s):", fim-inicio, params_updated)


def Inference(param_model, lst_features_test, 
              lst_distancias_test, lst_mol_sizes_test):
   
   
   lst_pdct_target = list()
   salvarParametros = tf.train.Saver()
   caminho_parametros = "./parameters/parameters.ckp"
   
   with tf.Session() as tf_sess:
    salvarParametros.restore(tf_sess, caminho_parametros)
    inicio = time.time()
    
    for idx in range(len(lst_mol_sizes_test)):
      batch_features = lst_features_test[idx]
      batch_dist = lst_distancias_test[idx]
      batch_features_padding = GetFeaturesPaddingAdjustments([lst_mol_sizes_test[idx]], max_num_atm)
      batch_cfconv_padding = GetDistPaddingAdjustments([lst_mol_sizes_test[idx]], max_num_atm)
      
      pdct = tf_sess.run(param_model, feed_dict = {plc_batch_features: batch_features,
                                                   plc_batch_dist: batch_dist,
                                                   plc_batch_bias: batch_features_padding,
                                                   plc_batch_bias_dist: batch_cfconv_padding})
      lst_pdct_target.append(pdct.flatten()[0])
    
    return lst_pdct_target
