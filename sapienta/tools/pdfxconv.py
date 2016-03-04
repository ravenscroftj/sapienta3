#!/usr/bin/env python
'''
This script enables the conversion of a PDF document to XML recognised by Sapienta via pdfx

'''
import time
import sys
import os
import logging
import signal
import traceback

from multiprocessing import Pool,Queue,current_process

import sapienta

from optparse import OptionParser
from sapienta.tools.converter import PDFXConverter
from sapienta.tools.annotate import Annotator


#from sapienta.tools.split import SentenceSplitter as Splitter
from sapienta.tools.sssplit import SSSplit as Splitter

#these are globals that are populated by each worker process
my_anno = None
my_pdfx = None
my_splitter = None
logger  = None
options = None
resultq = None

def init_worker(q, rq=None):
    global my_anno, my_pdfx, logger, my_splitter, resultq


    signal.signal(signal.SIGINT, signal.SIG_IGN)

    i,config = q.get()
    resultq = rq
    logger = logging.getLogger("pdfxconv:worker%d" % i )

    logger.info("Initializing PDF splitter")
    my_pdfx = PDFXConverter()

    logger.info("Initialising sentence splitter")
    my_splitter = Splitter()

    logger.info("Initialising annotator %d", i)
    my_anno = Annotator(config=config, logger=logger)

    logger.info("Using C&C server %s", config['SAPIENTA_CANDC_SOAP_LOCATION'])


def anno_e(work):
    global logger

    errors = 0
    while errors < 10:
        try:
            annotate(work)
            return
        except Exception, e:
            logger.error(e)
            logger.error(traceback.format_exc())
            errors += 1
            logger.info("Attempt %i of 10 Sleeping for 5 secs after encountering an error...", errors+1)
            time.sleep(5)
    logger.error("Giving up on %s", work[0])

def annotate(work):
    global my_anno, my_pdfx, my_splitter, logger, resultq

    infile, options = work

    name,ext = os.path.splitext(infile)

    my_anno.marginal = options.marginal

    bmk = {}

    if not(os.path.exists(infile)):
        logger.warning("Input file %s does not exist", infile)
        return

    if options.benchmark:
        bmk['paper'] = name
        bmk['start'] = time.clock()
        bmk['size']  = os.path.getsize(infile)



    outfile = name + ".xml"


    #annotation requires splitting
    if(options.annotate and not options.nosplit):
        options.split = True

            
    if(ext == ".pdf"):

        logging.info("Converting %s", infile)


        if options.benchmark:
            bmk['pdfx_start'] = time.clock()
       
        my_pdfx.convert(infile, outfile)
        split_infile = outfile

        if options.benchmark:
            bmk['pdfx_stop'] = time.clock()
       

        
    elif( ext == ".xml"):
        logging.info("No conversion needed on %s", infile)
        split_infile = infile

    else:
        logging.info("Unrecognised format for file %s", infile)
        return

    if(options.split):
        logging.info("Splitting sentences in %s", infile)

        outfile = name + "_split.xml"
        
        if options.benchmark:
            bmk['split_start'] = time.clock()
       
        print "Pretty mode is ", options.pretty
        

        my_splitter.split(split_infile, outfile, 
                pp=options.pretty, mode2=options.mode2)


        if options.benchmark:
            bmk['split_stop'] = time.clock()
       

    anno_infile = outfile

    if(options.annotate):

        #build annotated filename
        outfile = name + "_annotated.xml"
        logging.info("Annotating file and saving to %s", outfile)


        if options.benchmark:
            bmk['anno_start'] = time.clock()

        my_anno.annotate( anno_infile, outfile )

        if options.benchmark:
            bmk['anno_stop'] = time.clock()

    if options.benchmark:
        resultq.put(bmk)




def main():
    
    usage = "usage: %prog [options] file1.pdf [file2.pdf] "
    
    parser = OptionParser(usage=usage)
    parser.add_option("-s", "--split-sent", action="store_true", dest="split",
        help="If true, split sentences using NLTK sentence splitter")

    parser.add_option("--marginal", dest="marginal", default=False,
            action="store_true", help="If annotating produce marginal labels in <filename>.marginal.txt")
    parser.add_option("--splitter", dest="splitter", default="sssplit", 
            help="Choose which sentence splitter to use [sssplit,textsentence]")
    parser.add_option("--nosplit",dest="nosplit",action="store_true",
            help="Turn off splitting when annotating (if your corpus is already split and you just want to annotate)")

    parser.add_option('--mode2', dest='mode2', default=False, action="store_true", help="If true then formats in a SAPIENTA compatible way.")

    parser.add_option('--no-whitespace', dest='pretty', default=True,action="store_false",
            help="If set then removes all whitespace from output document")

    parser.add_option("-a", "--annotate", action="store_true", dest="annotate",
        help="If true, annotate the sentence-split XML with CoreSC labels")
    parser.add_option("-v", "--verbose", action="store_true", dest="verbose",
        help="If set, provides more verbose output")
    parser.add_option("--blacklist", help="Add elements to blacklisted splitter elements", 
            dest="extra_blacklist", default="")

    parser.add_option("--benchmark", help="Record results of annotations", dest="benchmark",
            action="store_true")

    parser.add_option("--parallel", help="Specify how many cpus to use", dest="cpus", type=int, default=1)

    parser.add_option("--candc-hosts", help="List of URLS that C&C is running on", dest="candc", default="http://localhost:9004/")
    
    (options, args) = parser.parse_args()

    candc_hosts = [ x for x in options.candc.split(",") if x.strip() != "" ]

    if(options.verbose):
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("Verbose mode on.")
    else:
        logging.basicConfig(level=logging.INFO)

    if( len(args) < 1):
        parser.print_help()
        sys.exit(1)

    if(options.annotate):
        a = Annotator()

    if options.cpus > 0:
        q = Queue()
        rq = Queue()
        for i in range(0, options.cpus):
            myconf = dict(**sapienta.app.config)

            if i < len(candc_hosts):
                myconf['SAPIENTA_CANDC_SOAP_LOCATION'] = candc_hosts[i]
            else:
                x = i % len(candc_hosts)
                myconf['SAPIENTA_CANDC_SOAP_LOCATION'] = candc_hosts[x]

            q.put((i,myconf))

        p = Pool(processes=options.cpus, initializer=init_worker, initargs=[q,rq])

        try:
            p.map(anno_e, [ (x,options) for x in args] )
        except KeyboardInterrupt:
            print "Killing workers"
            p.terminate()
            p.join()

    if options.benchmark:
        logging.info("Storing benchmark data")
        import csv

        with open("benchmark.csv", "wb") as f:
            w = csv.writer(f)
            w.writerow(["file","size", "pdftime", "splittime", "annotime"])

            while not rq.empty():
                bmk = rq.get()
    
                result = [bmk['paper'], bmk['size']]

                if 'pdfx_start' in bmk:
                    result.append(bmk['pdfx_stop'] - bmk['pdfx_start'])
                else:
                    result.append(0)


                if 'split_start' in bmk:
                    result.append(bmk['split_stop'] - bmk['split_start'])
                else:
                    result.append(0)


                if 'anno_start' in bmk:
                    result.append(bmk['anno_stop'] - bmk['anno_start'])
                else:
                    result.append(0)

                w.writerow(result)

if __name__ == "__main__":
    main()
