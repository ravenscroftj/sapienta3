'''
Created on 28 Feb 2016

@author: James Ravenscorft
'''
import logging
import os
import pickle
import spacy

import numpy as np

from gensim.models.word2vec import Word2Vec

from sapienta.ml.nnet import SapientaNeuralNet, CONTEXT_WINDOW, WORDVEC_SIZE, ALL_CORESCS

class FeatureExtractorBase:
    """This class has some reusable functions for extracting features
    """

    def __init__(self, modelFile, cacheDir, logger=None):
    
        self.modelFile = modelFile
        self.cacheDir = cacheDir

        if logger == None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        self.wv = None
            
    #------------------------------------------------------------------------------------------------
    
    def extractFeatures(self, file, cache=True):
        """Extract features from the given file and cache them"""

        from sapienta.ml.docparser import SciXML
        #from sapienta.ml.candc import SoapClient

        cachedName = os.path.join(self.cacheDir, os.path.basename(file))

        if os.path.exists(cachedName):

            self.logger.info("Loading features from %s", cachedName)
            with open(cachedName, 'rb') as f:
                features = pickle.load(f)
            return features

        else:
            self.logger.info("Generating features for %s", file)

            parser = SciXML()
            doc = parser.parse(file)
            #candcClient = SoapClient(self.config)
            processedSentences = []


            for sentence in doc.yieldSentences():
                #candcFeatures = candcClient.getFeatures(sentence.content)
                #sentence.candcFeatures = candcFeatures
                processedSentences.append(sentence)

                sentence.wordvectors = []
                
                for word in sentence.content.split(" "):
                    
                    if self.wv == None:
                        self.load_word_vectors()
                    
                    if word in self.wv.vocab:

                        sentence.wordvectors.append(self.wv[word])
            
            nnet_input = self._doc2vec(processedSentences)
            
            if cache:
                self.logger.debug("Caching features at %s", cachedName)

                with open(cachedName, 'wb') as f:
                    pickle.dump(nnet_input, f, -1)

            return nnet_input
        
    def gen_context_window(self, wordvectors):
        
        all_context = np.zeros( (len(wordvectors), CONTEXT_WINDOW,WORDVEC_SIZE) )
        
        half_window = CONTEXT_WINDOW // 2
        
        wv = [np.zeros(WORDVEC_SIZE)] * half_window + wordvectors + [np.zeros(WORDVEC_SIZE)] * half_window
        
        for i in range(0, len(wordvectors)):
            all_context[i] = np.array( wv[ i: i+CONTEXT_WINDOW ] )
        
    def _doc2output(self, sentences):
        
        docsize = len(sentences)
        
        blob = np.zeros( (docsize, len(ALL_CORESCS) ) )
        
        for i,x in enumerate([ sentence.corescLabel for sentence in sentences]):
            blob[i, ALL_CORESCS.index(x)] = 1
        
                
    def _doc2vec(self, sentences):
        
        batchsize = len(sentences)
        seqlen = max([len(x.wordvectors) for x in sentences])
        
        blob = np.zeros((batchsize, seqlen, CONTEXT_WINDOW, WORDVEC_SIZE))
        mask = np.zeros((batchsize,seqlen))
        
        for i, x in enumerate(sentences):
            slen = len(x.wordvectors)
            blob[i, :slen ] = self.gen_context_window(x.wordvectors)
            mask[i, :slen] = 1
        
        return blob, mask, self._doc2output(sentences)
        
    def load_word_vectors(self):
        self.wv= Word2Vec.load_word2vec_format("/home/james/workspace/sapienta3/GoogleNews-vectors-negative300.bin.gz", binary=True)


class FeatureFile:
    """Simple data structure that provides lazy-loading for training nnet"""
    name = None
    trainer = None
    
    def __init__(self, name, trainer):
        self.name = name
        self.trainer = trainer
    
    def __enter__(self):
        self.input, self.mask, self.output = self.trainer.extractFeatures(self.name)
        return self
        
    def __exit__(self, type, value, tb):
        self.input = None
        self.mask = None
        self.output = None

        
#-----------------------------------------------------------------------------

class SAPIENTATrainer(FeatureExtractorBase):
    """Base class for training systems and preprocessing features"""
    def __init__(self, modelFile, cacheDir, logger=None):
        """Create a sapienta trainer object"""

        FeatureExtractorBase.__init__(self, modelFile, cacheDir, logger)

        if not os.path.exists(cacheDir):
            os.makedirs(cacheDir)


    #------------------------------------------------------------------------------------------------

    def preprocess(self, filenames):
        """Extract initial features from the files and collect word vectors"""


    #------------------------------------------------------------------------------------------------

    def train(self, trainfiles):
        """Stub overriden by concrete implementations to do preprocessing and training
        """
        raise Exception("Not implemented! Use a subclass of SAPIENTATrainer")

    #------------------------------------------------------------------------------------------------
    
    def test(self, testFiles):
        """Stub overriden by concrete implementations to do testing of model
        """
        raise Exception("Not implemented! Use a subclass of SAPIENTATrainer")
    

#--------------------------------------------------------------------------------------------------

class NNetTrainer(SAPIENTATrainer):
    """Specific implementation of SAPIENTA trainer that uses CRFSuite for training"""

    def train(self, trainfiles):
        self.preprocess(trainfiles)
        
        net = SapientaNeuralNet(self.logger)
        
        #self.trainCRF(trainfiles)
        for file in trainfiles:
                
            #for sent in sentences:
            net.add_training_doc(FeatureFile(file, self))      
            
        net.train()          

    #------------------------------------------------------------------------------------------------
    