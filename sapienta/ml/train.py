'''
Created on 28 Feb 2016

@author: James Ravenscorft
'''
import logging
import os
import pickle
import spacy
import csv

import numpy as np

from gensim.models.word2vec import Word2Vec

from sapienta.ml.nnet import SapientaNeuralNet, CONTEXT_WINDOW, WORDVEC_SIZE, ALL_CORESCS

from sapienta.ml.wordserve import WordservClient
from collections import Counter

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

        self.wv = WordservClient("localhost",5000)
            
    #------------------------------------------------------------------------------------------------
    
    def extractFeatures(self, file, cache=True):
        """Extract features from the given file and cache them"""

        from sapienta.ml.docparser import SciXML
        #from sapienta.ml.candc import SoapClient

        cachedName = os.path.join(self.cacheDir, os.path.basename(file))

        if os.path.exists(cachedName):

            self.logger.debug("Loading features from %s", cachedName)
            with open(cachedName, 'rb') as f:
                features = pickle.load(f)
            return features

        else:
            self.logger.debug("Generating features for %s", file)

            parser = SciXML()
            doc = parser.parse(file)
            #candcClient = SoapClient(self.config)
            processedSentences = []


            for sentence in doc.yieldSentences():
                #candcFeatures = candcClient.getFeatures(sentence.content)
                #sentence.candcFeatures = candcFeatures
                processedSentences.append(sentence)

                sentence.wordvectors = []
                
                words = sentence.content.split(" ")
                
                vecs = self.wv.vector(words)
                
                sentence.wordvectors = [ vecs[w] for w in words ]
            
            nnet_input = self._doc2vec(processedSentences)
            
            if cache:
                self.logger.debug("Caching features at %s", cachedName)

                with open(cachedName, 'wb') as f:
                    pickle.dump(nnet_input, f, -1)

            return nnet_input
        
    #def gen_context_window(self, wordvectors):
    #    
    #    all_context = np.zeros( (len(wordvectors) * CONTEXT_WINDOW, WORDVEC_SIZE) )
    #    
    #    half_window = CONTEXT_WINDOW // 2
    #    
    #    wv = [np.zeros(WORDVEC_SIZE)] * half_window + wordvectors + [np.zeros(WORDVEC_SIZE)] * half_window
    #    
    #    for i in range(0, len(wordvectors)):
    #        all_context[i] = np.array( wv[ i: i+CONTEXT_WINDOW ] )
        
    def _doc2output(self, sentences):
        
        docsize = len(sentences)
        
        blob = np.zeros( (docsize, len(ALL_CORESCS) ) )
        
        for i,x in enumerate([ sentence.corescLabel for sentence in sentences]):
            blob[i, ALL_CORESCS.index(x)] = 1
            
        return blob
        
                
    def _doc2vec(self, sentences):
        
        batchsize = len(sentences)
        seqlen = max([len(x.wordvectors) for x in sentences])
        
        blob = np.zeros((batchsize, seqlen, WORDVEC_SIZE))
        mask = np.zeros((batchsize,seqlen))
        
        for i, x in enumerate(sentences):
            slen = len(x.wordvectors)
            blob[i, :slen ] = x.wordvectors
            mask[i, :slen] = 1
        
        return blob, mask, self._doc2output(sentences)
        
    def load_word_vectors(self):
        self.wv= Word2Vec.load_word2vec_format("/opt/GoogleNews-vectors-negative300.bin.gz", binary=True)


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

    def train(self, trainfiles, num_epochs=100):
        self.preprocess(trainfiles)
        
        net = SapientaNeuralNet(self.logger)
        
        #self.trainCRF(trainfiles)
        for file in trainfiles:
                
            #for sent in sentences:
            net.add_training_doc(FeatureFile(file, self))      
            
        net.train(self.modelFile, num_epochs=num_epochs)          

    #------------------------------------------------------------------------------------------------
    
    def test(self, testfiles):
                
        net = SapientaNeuralNet(self.logger)
        net.load(self.modelFile)
        net.compile()
        
        correct = 0
        all = 0
        
        predLabels = []
        trueLabels = []
        
        
        for file in testfiles:
            with FeatureFile(file,self) as f:
                
                for i,sentence in enumerate(net.label(f.input, f.mask)):
                    label = (max(enumerate(sentence), key=lambda x:x[1]))
                    actual = (max(enumerate(f.output[i]), key=lambda x:x[1]))
                    
                    predLabels.append(ALL_CORESCS[label[0]])
                    trueLabels.append(ALL_CORESCS[actual[0]])
                    
                    self.logger.debug("%s With %d%% confidence", ALL_CORESCS[label[0]], label[1]*100)
                    self.logger.debug("Actual label is %s", ALL_CORESCS[actual[0]])
            
                    all += 1
                    
                    if actual[0] == label[0]:
                        correct += 1
                    
                    
        self.logger.info("Got %d out of %d correct (That's %f%%)", correct,all, correct/all*100)
        
        return trueLabels, predLabels
        
        
    def calcPrecRecall(self, trueLabels, predictedLabels):
        
        tp = Counter()
        fp = Counter()
        fn = Counter()
        
        for true, predicted in zip(trueLabels,predictedLabels):
            if true == predicted:
                tp[true] += 1
            else:
                fp[predicted] += 1
                fn[true] += 1
                
        measures = {}
        for label in ALL_CORESCS:
            if tp[label] == 0:
                prec = 0
                rec = 0
            else:
                prec = tp[label] / (tp[label] + fp[label])
                rec  = tp[label] / (tp[label] + fn[label])  
                
            if (prec + rec) > 0:
                fm = (2 * prec * rec ) / (prec + rec)
            else:
                fm = 0
            
            measures[label] = (prec,rec,fm, tp[label], fp[label], fn[label])
        
        return measures        
        
    #------------------------------------------------------------------------------------------------
    
    def writePrecRecall(self, filename, trueLabels, predictedLabels):
        
        with open(filename,"w") as f:
            
            csvw = csv.writer(f)
            # write header
            csvw.writerow(['Label','Precision','Recall','F-Measure','True Positive', ' False Positive', 'False Negative'])
            #calculate the measurements
            measures = self.calcPrecRecall(trueLabels, predictedLabels)
            #write measurements to disk
            for label in measures:
                csvw.writerow( [label] + list(measures[label]))
        
        
                    