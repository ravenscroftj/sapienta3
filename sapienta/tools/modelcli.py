#!/usr/local/bin/python2.7
# encoding: utf-8
'''
sapienta.tools.modelcli -- train neural network models for SAPIENTA

sapienta.tools.modelcli allows users to train and manage neural net models from the commandline

@author:     James Ravenscroft

@copyright:  2016 Warwick Institute for Science of Cities. All rights reserved.

@license:    MIT

@contact:    j.ravenscroft@warwick.ac.uk
@deffield    updated: Updated
'''

import sys
import os
import logging

from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
from sapienta.ml.folds import get_folds

__all__ = []
__version__ = 0.1
__date__ = '2016-02-28'
__updated__ = '2016-02-28'

DEBUG = 1
TESTRUN = 0
PROFILE = 0

CACHEDIR = "cachedFeatures"

class CLIError(Exception):
    '''Generic exception to raise and log different fatal errors.'''
    def __init__(self, msg):
        super(CLIError).__init__(type(self))
        self.msg = "E: %s" % msg
    def __str__(self):
        return self.msg
    def __unicode__(self):
        return self.msg
    
    
def generateFolds(corpusDir, foldsFile, logger=None):
    
    if logger == None:
        logger = logging.getLogger(__name__)
    
    def genFileName(x):
        if x['annotator'] != "":
            return os.path.join(corpusDir, x['filename'] + 
                                "_mode2." + x['annotator'] + ".xml") 
        else:
            return os.path.join(corpusDir, x['filename'] + 
                                "_mode2.xml")


    folds = get_folds(foldsFile)

    allFiles =  [f for f in [ genFileName(fdict) 
                for x in folds for fdict in x ] 
                        if os.path.exists(f)]
    
    fixtures = []
    
    for f, fold in enumerate(folds):

        testFiles = []
        sents = 0
        
        for filedict in fold:
            fname = genFileName(filedict)
    
            sents += int(filedict['total_sentence'])
    
            if not os.path.isfile(fname):
                logger.warn("No file %s detected.", fname)
            else:
                testFiles.append(fname)
    
        logger.info("Fold %d has %d files and %d sentences total" + 
                " (which will be excluded)", f, len(testFiles), sents)
    
        #calculate which files to use for training
        trainFiles = [file for file in allFiles if file not in testFiles]
    
        fixtures.append( (testFiles, trainFiles) )
        
    return fixtures    

def main(argv=None): # IGNORE:C0111
    '''Command line options.'''

    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    program_name = os.path.basename(sys.argv[0])
    program_version = "v%s" % __version__
    program_build_date = str(__updated__)
    program_version_message = '%%(prog)s %s (%s)' % (program_version, program_build_date)
    program_shortdesc = __import__('__main__').__doc__.split("\n")[1]
    program_license = '''%s

  Created by user_name on %s.
  Copyright 2016 organization_name. All rights reserved.

  Licensed under the Apache License 2.0
  http://www.apache.org/licenses/LICENSE-2.0

  Distributed on an "AS IS" basis without warranties
  or conditions of any kind, either express or implied.

USAGE
''' % (program_shortdesc, __date__)

    try:
        # Setup argument parser
        parser = ArgumentParser(description=program_license, formatter_class=RawDescriptionHelpFormatter)
        
        parser.add_argument("action", choices=['train','test','crossval'])
        
        parser.add_argument("-m", "--model_name", action="store", help="Set the file name of the model (ignored in crossval mode)")
        parser.add_argument("-f", "--folds_file", action="store", help="Name of folds CSV file (ignored unless running in crossval mode)")
        parser.add_argument("corpus_dir", action="store", help="Name of corpus directory to traverse")
        parser.add_argument("-e", "--epochs", action="store", type=int, default=100, help="Set the number of epochs to train for, default 100")
        
        parser.add_argument("-r","--resultsfile", action="store", dest="results_file", default="results.csv", 
                            help="In test mode name of CSV to dump results to. In crossval, name template to which the fold number is prepended. Defaults to results.csv")
        
        
        #parser.add_argument(dest="paths", help="paths to folder(s) with source file(s) [default: %(default)s]", metavar="path", nargs='+')
        
        parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="set verbosity level [default: %(default)s]")
        parser.add_argument('-V', '--version', action='version', version=program_version_message)
        

        # Process arguments
        args = parser.parse_args()

        if args.verbose:
            logging.basicConfig(level=logging.DEBUG)
            logging.debug("Verbose mode on")
        else:
            logging.basicConfig(level=logging.INFO)
            
        logging.getLogger("requests").setLevel(logging.WARNING)

            
        if args.corpus_dir == None:
            logging.error("You must provide a valid corpus directory for training")
            return 3
        
        if not os.path.isdir(args.corpus_dir):
            logging.error("Invalid corpus directory %s", args.corpus_dir)
            return 3
        
        all_files = []
        for root, _, files in os.walk(args.corpus_dir):
            
            if root.endswith(CACHEDIR):
                logging.debug("Skipping cache directory")
                continue
            
            for file in files:
                if file.endswith(".xml"):
                    all_files.append(os.path.join(root,file))
                    
        from sapienta.ml.train import NNetTrainer
        
        
        
        if args.action == "train":
            
            trainer = NNetTrainer(args.model_name, os.path.join(args.corpus_dir, CACHEDIR))
            
            if args.model_name == None:
                logging.error("Must specify name of model file to save trained model to")
                return -1
            
            if os.path.exists(args.model_name):
                logging.error("The provided model path '%s' exists. Refusing to overwrite (rename it out of the way)", args.model_name)
                return -1
            
            trainer.train(all_files, num_epochs=args.epochs)
            
        if args.action == "test":
            
            trainer = NNetTrainer(args.model_name, os.path.join(args.corpus_dir, CACHEDIR))
            
            if args.model_name == None:
                logging.error("Must specify name of model file to test")
                return -1
            
            if not os.path.exists(args.model_name):
                logging.error("The provided model path '%s' does not exist", args.model_name)
                return -1
            
            
            trueLabels,predictedLabels = trainer.test(all_files)
            trainer.writePrecRecall(args.results_file, trueLabels, predictedLabels)
            
        if args.action == "crossval":
            
            if args.folds_file == None:
                logging.error("You must provide a folds file to train against")
                return -1
            
            allTrue, allPredicted = [],[]
            
            for i, (testFiles, trainFiles) in enumerate(generateFolds(args.corpus_dir, args.folds_file)):
                
                
                model_file = os.path.join(args.corpus_dir, "model_fold_{}.npz".format(i) )
                results_file = os.path.join(args.corpus_dir,"results_fold_{}.csv".format(i))
                
                trainer = NNetTrainer(model_file, os.path.join(args.corpus_dir, CACHEDIR))
                
                
                if os.path.exists(model_file):
                    logging.warn("The model file %s already exists, refusing to overwrite. Skipping to test phase.", model_file)
                else:
                    trainer.train(trainFiles, args.epochs)
                    
                trueLabels,predictedLabels = trainer.test(testFiles)
                
                allTrue += trueLabels
                allPredicted += predictedLabels
                
                trainer.writePrecRecall(results_file, trueLabels, predictedLabels)
                
            
            results_file = os.path.join(args.corpus_dir,"results_micro_all.csv")      
            trainer.writePrecRecall(results_file, allTrue, allPredicted)          

        return 0
    
    
    
    except KeyboardInterrupt:
        ### handle keyboard interrupt ###
        return 0
    except Exception as e:
        if DEBUG or TESTRUN:
            raise(e)
        indent = len(program_name) * " "
        sys.stderr.write(program_name + ": " + repr(e) + "\n")
        sys.stderr.write(indent + "  for help use --help")
        return 2

if __name__ == "__main__":
    #if DEBUG:
        #sys.argv.append("-h")
        #sys.argv.append("-v")
        #sys.argv.append("-r")
    if TESTRUN:
        import doctest
        doctest.testmod()
    if PROFILE:
        import cProfile
        import pstats
        profile_filename = 'sapienta.ml.train_profile.txt'
        cProfile.run('main()', profile_filename)
        statsfile = open("profile_stats.txt", "wb")
        p = pstats.Stats(profile_filename, stream=statsfile)
        stats = p.strip_dirs().sort_stats('cumulative')
        stats.print_stats()
        statsfile.close()
        sys.exit(0)
    sys.exit(main())