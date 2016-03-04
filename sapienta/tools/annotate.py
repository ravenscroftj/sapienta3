'''

Take a known, split paper and annotate it using Sapienta remotely

'''
import sys
import os
import pycurl
import codecs
import subprocess
import tempfile
import logging

from urllib import urlencode
from progressbar import ProgressBar
from xml.dom import minidom
from curlutil import CURLUploader
from sapienta import app

config = app.config

from collections import Counter



class SapientaException(Exception):
    pass


class BaseAnnotator(object):
    """Basic annotator with some boilerplate stuff in it"""


    marginal = False

    def _annotateXML(self, labels):
        """Read in the xml document and add labels to it
        """

        c = Counter()

        for s in self.doc.getElementsByTagName("s"):
            if s.parentNode == None or s.parentNode.localName == "article-title": continue 

            sid = s.getAttribute("sid")

            if labels.has_key(sid):

                label = labels[sid]

                c[label] += 1

                annoEl = self.doc.createElement("CoreSc1")
                annoEl.setAttribute("type", label)
                annoEl.setAttribute("conceptID", label + str(c[label]))
                annoEl.setAttribute("novelty", "None")
                annoEl.setAttribute("advantage","None")

                textEl = self.doc.createElement("text")

                children = s.childNodes

                # We're going to copy the current children somewhere for safety
                # then delete them, then add new CoreSc child and then put back 
                # the original children. The children will be wrapped by a 
                # new <text> element

                # So first, the copying.
                for ch in children:
                    clonech = ch.cloneNode(True)
                    textEl.appendChild(clonech)

                
                # Something odd about removing the children through a map or for-loop
                # meant that not all were being removed, hence this while-loop
                # Was previously map(s.removeChild, children)
                while s.hasChildNodes():
                    ch = s.firstChild
                    s.removeChild(ch)
                    ch.unlink()

                # Add the new annotation, and then the original children
                s.appendChild(annoEl)
                s.appendChild(textEl)




    def __upgradeXML(self):
        """When passed in an old annotation format doc, upgrade the format
        
        This method takes an XML document and replaces old fashioned 
        annotationART style elements with the new CoreSC style.
        """

        for annoEl in self.doc.getElementsByTagName("annotationART"):
            sentEl = annoEl.parentNode

            #remove annotation element from tree
            sentEl.removeChild(annoEl)

            #create the CoreSC element
            coreEl = self.doc.createElement("CoreSc1")


            for key in ['type', 'advantage', 'conceptID', 'novelty']:
                coreEl.setAttribute(key, annoEl.getAttribute(key))

            for child in annoEl.childNodes:
                sentEl.appendChild(child.cloneNode(True))
                child.unlink()

            sentEl.insertBefore( coreEl, sentEl.firstChild)



    def annotate(self, infile, outfile):
        """Do the actual annotation work"""

        #parse doc to see if annotations already present
        with open(infile,"rb") as f:
            self.doc = minidom.parse(f)

        if len(self.doc.getElementsByTagName("annotationART")) > 0:
            self.__upgradeXML()




#MODEL_PATH = str(os.path.join(config['MODELS_DIR'], "a.model"))
class LocalPythonAnnotator(BaseAnnotator):

    def __init__(self,config=config,logger=None):
        from sapienta.ml.annotate import CRFAnnotator
        from tempfile import mkdtemp

        cacheDir = mkdtemp()
        modelFile = config['SAPIENTA_MODEL_FILE']
        ngramsFile = config['SAPIENTA_NGRAMS_FILE']

        features = ['ngrams', 'verbs', 'verbclass','verbpos', 'passive','triples','relations','positions' ]


        self.annotator = CRFAnnotator(modelFile, ngramsFile, cacheDir, features, config, logger)

    #------------------------------------------------------------------------- 
    def annotate(self, filename, outfilename):
        """Use Python version of sapienta to annotate the paper"""
        BaseAnnotator.annotate(self, filename, outfilename)
        labels = self.annotator.annotate(filename, self.marginal)


        self._annotateXML(labels)

        with codecs.open(outfilename,'w', encoding='utf-8') as f:
            self.doc.writexml(f)


    #-------------------------------------------------------------------------

class LocalPerlAnnotator(BaseAnnotator):
    """Uses Perl version of sapienta to annotate the given paper"""

    def __init__(self):

        self.perldir = config['SAPIENTA_PERL_DIR']
        self.resultdir = config['SAPIENTA_RESULT_DIR']


    def annotate(self,infile,outfile):
        """Start a local instance of Sapienta and annotate the file"""

        #call parent annotation class
        BaseAnnotator.annotate(self,infile,outfile)

        logging.info("Running local Perl annotator using SAPIENTA pipeline")

        #parse doc to see if annotations already present
        if len(self.doc.getElementsByTagName("CoreSc1")) < 1:
            args = ["perl", "pipeline_for_sapient_crfsuite.perl", 
                os.path.abspath(infile)]

            f = tempfile.TemporaryFile()


            logging.info("Calling %s", str(args))
            p = subprocess.Popen(args, cwd=self.perldir, 
                stdout=f, stderr=f)
            logging.info("Waiting for SAPIENTA...")
            p.wait()
            logging.info("SAPIENTA is finished...")

            f.close()

            #now open the results file and recombine
            result = os.path.join(self.resultdir, os.path.basename(infile), "result.txt")

            with open(result,'r') as f:
                labels = f.read().split(">")

            self._annotateXML(labels)

            #delete the result file
            os.unlink(result)

        with codecs.open(outfile,'w', encoding='utf-8') as f:
            self.doc.writexml(f)

                


#------------------------------------------------------------------            

SAPIENTA_URL="http://www.ebi.ac.uk/Rebholz-srv/sapienta/CoreSCWeb/submitRPC"

class RemoteAnnotator(BaseAnnotator, CURLUploader):
    """Class that submits a remote annotation job to sapienta servers and saves
    results
    """

    def __init__(self):
        """"""

    def annotate(self, infile, outfile):
        """Do the actual annotation work"""

        BaseAnnotator.annotate(self, infile, outfile)

        #parse doc to see if annotations already present
        if len(self.doc.getElementsByTagName("CoreSc1")) < 1:

            pdata = [('paper', (pycurl.FORM_FILE, infile) )]

            c = pycurl.Curl()
            c.setopt(pycurl.URL, SAPIENTA_URL)
            c.setopt(pycurl.POST,1)
            c.setopt(pycurl.HTTPPOST, pdata)

            logging.info("Uploading %s to annotation server", infile)

            self.perform(c)
            try:
                tmpnam, sents = self.result.split(":")
            except Exception as e:
                raise SapientaException("Empty response from SAPIENTA")

            labels = sents.split(">")

            self._annotateXML( labels )

        with codecs.open(outfile,'w', encoding='utf-8') as f:
            self.doc.writexml(f)






#Set annotator to remote annotator for now

if config.has_key('SAPIENTA_ANNOTATE_METHOD'):
    
    if config['SAPIENTA_ANNOTATE_METHOD'] == 'PERL':
        Annotator = LocalPerlAnnotator
    else:
    #elif config['SAPIENTA_ANNOTATE_METHOD'] == 'PYTHON':
        Annotator = LocalPythonAnnotator
else:
    Annotator = LocalPythonAnnotator


if __name__ == "__main__":
    r = Annotator()

    for file in sys.argv[1:]:
        root,ext = os.path.splitext(file)
        r.annotate(file, root+".annotated.xml")

