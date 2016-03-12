'''
Created on 28 Feb 2016

@author: James Ravenscroft
'''
import random
import theano
import lasagne
import logging
import theano.tensor as T
import numpy as np

from lasagne.layers import *
from lasagne import layers

CONTEXT_WINDOW = 5

WORDVEC_SIZE = 300

LEARNING_RATE = 0.01

ALL_CORESCS = ['Bac', 'Con', 'Exp', 'Goa', 'Hyp', 'Met', 'Mod', 'Mot', 'Obj', 'Obs', 'Res', '']


class SapientaNeuralNet(object):
    '''
    classdocs
    '''

    docs = {}
    
    _compiled = False

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
        self.in_layer = InputLayer((None,None,WORDVEC_SIZE))

        self.i_mask = InputLayer((None,None))

        batchsize,seqlen, _ = self.in_layer.input_var.shape

        #word -> sentence embedding layer
        l_lstm1 = LSTMLayer(self.in_layer, 
                            num_units=100, 
                            mask_input=self.i_mask, 
                            only_return_final=True,
                            nonlinearity=lasagne.nonlinearities.tanh)
        
        self.lstm = l_lstm1
        
        self.logger.debug("LSTM layer shape: %s",lasagne.layers.get_output_shape(l_lstm1))
        
        l_dropout1 = DropoutLayer(l_lstm1)
        
        l_shp1 = ReshapeLayer(l_dropout1, (1, batchsize, 100))
        
        l_lstm2 = LSTMLayer(l_shp1, num_units=50)

        l_shp2 = ReshapeLayer(l_lstm2, (batchsize,50))
        
        #self.l_shp1 = l_shp1

        l_dense1 = DenseLayer(l_shp2, num_units=len(ALL_CORESCS), nonlinearity=lasagne.nonlinearities.softmax)
        
        self.dense = l_dense1

        self.l_out = self.dense #ReshapeLayer(l_dense1, (batchsize, seqlen, len(ALL_CORESCS)))

        self.logger.debug("Output layer shape: %s", lasagne.layers.get_output_shape(self.l_out))
        
        
    def compile(self):
        """Builds models and functions for network on graphics card"""

        target_values = T.fmatrix('target_values')

        train_output = lasagne.layers.get_output(self.l_out)

        pred_output = lasagne.layers.get_output(self.l_out, deterministic=True)

        reg = lasagne.regularization.regularize_layer_params(self.l_out, lasagne.regularization.l2)

        train_cost = self.cost_function(train_output,target_values).mean() + reg * 0.01
        
        real_cost = self.cost_function(pred_output, target_values).mean()  + reg * 0.01

        # Retrieve all parameters from the network
        all_params = lasagne.layers.get_all_params(self.l_out,trainable=True)

        self.logger.info("Computing updates...")
        updates = self.update_function(train_cost, all_params, LEARNING_RATE)

        
        self.logger.info("Compiling functions ...")

        # compile Theano GPU functions for training and computing train_cost
        self._train = theano.function([self.in_layer.input_var, self.i_mask.input_var, target_values], train_cost, 
                updates=updates, allow_input_downcast=True)

        self._compute_cost = theano.function([self.in_layer.input_var, self.i_mask.input_var, target_values], real_cost, 
                allow_input_downcast=True)

        self.label = theano.function([self.in_layer.input_var, self.i_mask.input_var],pred_output,allow_input_downcast=True)
        
        self._compiled = True

        
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
        
    def train(self, modelfile, num_epochs=100):
        
        if not self._compiled:
            self.compile()
            
        self.logger.info("Starting training")
        
        docnames = list(self.docs.keys())
        
        valdoc = self.docs[docnames[0]]
        traindocs = docnames[1:]
        
        for i in range(0,num_epochs):
            
            #shuffle the documents for training each iteration
            self.logger.info("Shuffling training docs for epoch %d", i)
            random.shuffle(traindocs)
            
            for j, docname in enumerate(traindocs):
                doc = self.docs[docname]
                self.logger.debug("%d).......   Loading file %s", j, doc.name)
                with doc as current_doc:
                    
                    self.logger.debug("Training with input %s", current_doc.input.shape )
                    self.logger.debug("Expected output shape is %s", current_doc.output.shape)
                    
                    lstm_shape = self.lstm.get_output_shape_for([current_doc.input.shape, current_doc.mask.shape])
                    self.logger.debug("LSTM output shape for input is %s", lstm_shape)
                    
                    #reshape_shape = self.l_shp1.get_output_shape_for(lstm_shape)
                    #self.logger.debug("Reshape output shape for input is %s", reshape_shape)
                    
                    #dense_shape = self.dense.get_output_shape_for(reshape_shape)
                    
                    #self.logger.debug("Dense output shape for input is %s", dense_shape)
                    
                    #self.logger.debug("Output layer shape is %s", self.l_out.get_output_shape_for(dense_shape))
                    
                    self._train(current_doc.input, current_doc.mask, current_doc.output)
            
            with valdoc as doc:
                error = self._compute_cost(doc.input, doc.mask, doc.output)
            
            self.logger.info("Epoch %d complete. Error is %s ", i, error)
        
        self.logger.info("Finished training. Writing model to %s", modelfile)
        self.save(modelfile)
    