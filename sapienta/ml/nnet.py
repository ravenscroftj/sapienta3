'''
Created on 28 Feb 2016

@author: James Ravenscroft
'''

import shelve
import theano
import lasagne
import logging
import theano.tensor as T
import numpy as np

from lasagne.layers import *

N_HIDDEN = 9

CONTEXT_WINDOW = 5

WORDVEC_SIZE = 300

LEARNING_RATE = 0.01

ALL_CORESCS = ['Bac', 'Con', 'Exp', 'Goa', 'Hyp', 'Met', 'Mod', 'Mot', 'Obj', 'Obs', 'Res', '']


class SapientaNeuralNet(object):
    '''
    classdocs
    '''

    docs = {}

    def __init__(self, logger=None):
        '''
        Constructor
        '''
        
        self.logger = logger or logging.getLogger(__name__)
        self.cost_function   = lasagne.objectives.categorical_crossentropy
        self.update_function = lasagne.updates.nesterov_momentum
        
        self.construct_network()
        
        
    def construct_network(self):
        
        #input layer
        self.in_layer = InputLayer((None,None,CONTEXT_WINDOW, WORDVEC_SIZE))

        self.i_mask = InputLayer((None,None))

        batchsize,seqlen, _, _ = self.in_layer.input_var.shape

        #word -> sentence embedding layer
        l_lstm1 = LSTMLayer(self.in_layer, 
                            num_units=N_HIDDEN, 
                            mask_input=self.i_mask, 
                            nonlinearity=lasagne.nonlinearities.tanh, 
                            only_return_final=True)

        self.logger.debug("LSTM layer shape: %s",lasagne.layers.get_output_shape(l_lstm1))
        

        l_dropout1 = DropoutLayer(l_lstm1)
        

        l_shp1 = ReshapeLayer(l_dropout1, (-1, N_HIDDEN))

        l_dense1 = DenseLayer(l_shp1, num_units=len(ALL_CORESCS), nonlinearity=lasagne.nonlinearities.softmax)

        self.l_out = ReshapeLayer(l_dense1, (batchsize, seqlen, len(ALL_CORESCS)))

        self.logger.debug("Output layer shape: %s", lasagne.layers.get_output_shape(self.l_out))
        
        
    def compile(self):
        """Builds models and functions for network on graphics card"""

        target_values = T.dtensor3('target_values')

        network_output = lasagne.layers.get_output(self.l_out)

        pred_output = lasagne.layers.get_output(self.l_out, deterministic=True)

        cost = self.cost_function(network_output,target_values).mean()

        # Retrieve all parameters from the network
        all_params = lasagne.layers.get_all_params(self.l_out,trainable=True)

        self.logger.info("Computing updates...")
        updates = self.update_function(cost, all_params, LEARNING_RATE)

        
        self.logger.info("Compiling functions ...")

        # compile Theano GPU functions for training and computing cost
        self._train = theano.function([self.in_layer.input_var, target_values, self.i_mask.input_var], cost, 
                updates=updates, allow_input_downcast=True)

        self._compute_cost = theano.function([self.in_layer.input_var, target_values, self.i_mask.input_var], cost, 
                allow_input_downcast=True)

        self._label = theano.function([self.in_layer.input_var, self.i_mask.input_var],pred_output,allow_input_downcast=True)

        
    def save(self, filename):  
        self.logger.info("Saving model to %s...", filename) 
        np.savez(filename, *lasagne.layers.get_all_param_values(self.l_out))
    
    def load(self, filename):
        
        self.logger.info("Loading model from %s...", filename)
        
        with np.load(filename) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]

        lasagne.layers.set_all_param_values(self.l_out, param_values)

    def add_training_doc(self, file):
        self.docs[file.name] = file
        
    def train(self, num_epochs=10):
        
        self.logger.info("Starting training")
        for i in range(0,num_epochs):
            
            for fname, doc in self.docs.items():
                self.logger.debug("Loading file %d", fname)
                with doc as current_doc:
                    self._train(current_doc.input, doc.mask, doc.output)

        

    