#import xml.etree.cElementTree as ET
import sys
import re
import lxml.etree as ET
import os


highLevelContainerElements = ["DIV", "sec", "section", "abstract", "ABSTRACT" "article-title"]
pLevelContainerElements = ["P", "p", "region"]
referenceElements = ["REF", 'xref', 'ref']
commonAbbreviations = ['Fig','Figs', 'Ltd', 'St', 'al', 'ca', 'vs', 'viz', 'prot', 'Co', 'Ltd', 'No', 'Chem']

#from sapienta.tools.mlsplit import text_to_features

def is_str(s):
    return isinstance(s,str) or isinstance(s,unicode)


class SSSplit:

    classifier = None

    def normalize_sents(self):
        for i,s in enumerate(self.root.iter("s")):

            #set sentence ID
            s.set("sid", str(i+1))

    def load_authors(self):
        """Parse authors from citations and add to do not split list"""

        self.authors = []

        for el in self.root.iter('AUTHOR'):
            self.authors.append(" ".join(list(el.itertext())))


    def split(self, filename, outname=None, pp=True, mode2=False, *args, **kwargs):
        tree = ET.parse(filename)
        self.root = tree.getroot()



        #load list of referenced authors for sentence splitting purposes
        self.load_authors()

        #first find and split ABSTRACT (SciXML special case p-level container)
        #for el in self.root.iter("ABSTRACT"):
        #    self.split_plevel_container(el)
            
        #find and split abstract (Pubmed DTD special case high level container)
        #for el in self.root.iter("abstract"):
        #    self.split_high_level_container(el)

        #now we handle remaining high level containers such as <DIV> or <sec>
        for container in highLevelContainerElements:
            for el in self.root.iter(container):
                if len(list(el.iterancestors(*referenceElements))) < 1:
                    self.split_high_level_container(el)

        #assign sentence ids
        self.normalize_sents()

        if mode2:

            docname = os.path.splitext(filename)[0]

            tree.docinfo.clear()
            print tree.docinfo.doctype

            paper = ET.Element('PAPER')
            mode2 = ET.SubElement(paper, 'mode2', 
                    name=docname,
                    hasDoc="yes",
                    version="597"
                    )

            paper.append(self.root)

            tree = ET.ElementTree(paper)


        if outname != None:
            tree.write(outname, pretty_print=pp)
        else:
            return ET.tostring(self.root, pretty_print=pp)

    def split_high_level_container(self, containerEl):
        """A high level container is a section or similar
        
        High-level containers are container elements that contain p-level
        containers and do not have text or sentences as direct descendents.
        
        Examples of high level containers are <DIV> in SciXML and <section> 
        in DoCo XML"""

        for containerType in pLevelContainerElements:
            for el in set(containerEl.findall(containerType)):
                self.split_plevel_container(el)

        contains_sub_elements = False
        for container in highLevelContainerElements:
            if len(containerEl.findall('.//%s' % container)) > 0:
                contains_sub_elements = True
                break

        if len(containerEl.findall('.//s')) < 1 and not contains_sub_elements:
            self.split_plevel_container(containerEl)

        

    def split_plevel_container(self, containerEl):
        """A p-level container is a paragraph or similar
        
        P-level containers are containers that can contain text nodes as direct
        descendents i.e. <P> in SCIXML or a <region> in DoCoXML.

        This method splits sentences contained in P level containers taking into
        account the presence of sub-elements such as <xref> tyoes
        """

        #get a list of all child elements in containerNode to analyse
        siblings = list(containerEl)

        #if the container element has text insert in front of other siblings
        if (containerEl.text != None) and (containerEl.text.strip() != ""):
            self.no_plevel_splits = False
            siblings = [ containerEl.text] + siblings

        #self.splitSentencesML(siblings, containerEl)
        self.splitSentences(siblings, containerEl)
        
    def splitSentencesML(self, nodeList, containerEl):
        """Use machine learning model to do sentence splitting"""
        
        tokens = []
        for node in nodeList:
            if is_str(node):
                tokens += node.strip().split(" ")
            else:
                tail = node.tail
                node.tail = None
                tokens.append(node)
                
                if tail != None:
                    tokens += tail.strip().split(" ")
        
        self.splitTokens(tokens)
        
        self.endPLevelContainer(containerEl)
        
                    
    def splitTokens(self, tokens):
        
        self.newNodeList = []
        self.newSentence = []
        
        for tok, features in zip(tokens, text_to_features(tokens)):
            
            self.newSentence.append(tok)
            
            if self.classifier.classify(features):
                self.endCurrentSentence()
                
        self.endCurrentSentence()
                

    def splitSentences(self, nodeList, containerEl):
        """This xml-aware method builds sentence lists using nodes"""

        #new node list is the list of <s> elements to be created
        self.newNodeList = []
        #newsentence is the accumulator for sentence elements and strings
        self.newSentence = []

        #first we walk through all nodes inside the container
        while len(nodeList) > 0:
            
            el = nodeList.pop(0)

            #if the node is a string (or unicode)
            #run text splitting routine on it
            if is_str(el):
                self.splitTextBlock(el)

            # if node is an element, append it to the current sentence
            else:
            
                #if the node is a <REF> and this is a new sentence, chances are
                #it should be appended to the previous sentence
                # e.g. "this is the end of my sentence. [1]" 
                if (len(self.newSentence) < 1 and el.tag in referenceElements
                    and len(self.newNodeList) > 0):

                    textProc = None
                    if el.tail != None:
                        textProc = el.tail
                        el.tail = None

                    self.newNodeList[-1].append(el)
                    
                    if textProc != None:
                        self.splitTextBlock(textProc)
                else:
                    self.newSentence.append(el)
                    if el.tail != None:
                        self.splitTextBlock(el.tail)
                        #now remove the 'old' tail since the new one will be appended
                        el.tail = None

        # when we run out of child nodes for p-level container we know
        # we're at the end of the current sentence 
        # (sentences don't cross <p></p> boundaries)
        self.endCurrentSentence()

        # now we can be confident that we're finished with this container
        # so we can generate final xml form
        self.endPLevelContainer(containerEl)

    def splitTextBlock(self, txt, beforeNode=None):
        
        # first, if the text starts with a capital letter and
        # current sentence is not empty -we got it wrong - 
        # end current sentence now.
        #if(re.match("^[\(\[]?[A-Z]", txt) and len(self.newSentence) > 0):
        #    self.endCurrentSentence()
        
        pattern = re.compile('(\.|\?|\!)(?=\s*[\[\(A-Z0-9$])|\.$')

        m = pattern.search(txt)
        last = 0


        while m != None:

            # assume that the punctuation matched is the end of the sentence
            # (until otherwise proven)
            endOfSent = True

            #get last word before full stop (if not full stop we don't care)
            lastmatch = re.search("[\(\[]?(\S+)\.$", txt[last:m.end()])
            
            if lastmatch != None:
                lastword = lastmatch.group(1)
            else:
                lastword = None
            

            #if the last word is a common abbreviation, skip end of sentence
            if lastword != None and lastword in commonAbbreviations:
                endOfSent = False
                
            #if the last word is a single letter then it is usually an initial
            
            if lastword != None and re.match("^[\(\[]?[A-Z]$", lastword): 
                
                sent = txt[last:m.end()]
                
                #find the last letter in the word before the last word
                j = sent.rfind(lastword) - 2 
                interesting = sent[j:j+1]
                
                if interesting not in "0123456789":
                    endOfSent = False
            
            if txt[last:m.end()].strip() != '':
                self.newSentence.append(txt[last:m.end()])
            
            last = m.end()


            #if the dot matches the end of a common abbreviation, skip end of sentence

            #if we match digits around a dot then its probably a number so skip
            if re.match("[0-9]?\.[0-9]", txt[m.start()-1:m.end()+1]):
                endOfSent = False
                
            #if we match lower case letters it could be an abbreviation like e.g. or i.e.
            if re.match("[a-z]\.[a-z]", txt[m.start()-1:m.end()+1]):
                endOfSent = False


            #check if we should be ending the sentence this time around the loop
            if endOfSent:
                self.endCurrentSentence()

            m = pattern.search(txt, last)

        #the remnants of the string are the beginning of the next sentence
        if txt[last:] != '':
            self.newSentence.append(txt[last:])

        # note: we don't end sentence by default at this point because this could
        # just be the end of the text block and the start of a reference or formatting


    def endCurrentSentence(self):
        """Ends the current sentence being accumulated
        """
        if self.newSentence != []:
            #print self.newSentence
            self.newNodeList.append(self.newSentence)
            self.newSentence = []

    def endPLevelContainer(self, pContainer):
        """Process updates/splits in the current p-level container"""
        #prune all children of p container
        pContainer.text = None
        for el in pContainer:
            pContainer.remove(el)

        #generate sentences and append to container
        prevSent = None
        for sent in self.newNodeList:
            prevSent = self.generateSentence(sent,pContainer, prevSent)
            


    def generateSentence(self, sent, parent, prevSent):
        """Takes a list of strings and elements and turn into an <s> element
        
        Using the element tree subelement factory, create a sentence from
        a list of str and legal descendents (i.e. xref, ref)
        """
        
        sentEl = ET.SubElement(parent, "s")

        prevEl = None
        refOnly = True
        
        for item in sent:
            #are we dealing with text (string or unicode)
            if is_str(item):
                #refOnly is no longer true because we found text
                refOnly = False
                #if prev item is not set this is the first text node in the sentence
                if prevEl == None:
                    if sentEl.text != None:
                        sentEl.text += item
                    else:
                        sentEl.text = item
                #if prev item is set, this will be tacked on as the 'tail'
                else:
                    if prevEl.tail != None:
                        prevEl.tail += item
                    else:
                        prevEl.tail = item

            #else we're dealing with an element not text
            else:
                prevEl = item
                sentEl.append(item)
            
                
        if (is_str(sent[0]) and sent[0][0].islower() and prevSent != None):
            for item in sentEl:
                prevSent.append(item)
            parent.remove(sentEl)
            return

        if refOnly and prevSent != None:
            parent.remove(sentEl)

            for item in sent:
                prevSent.append(item)

            return prevSent
        else:
            return sentEl

    


if __name__ == "__main__":
    splitter = SSSplit()
    print splitter.split("/home/james/tmp/papers_for_type/research/journal.pbio.0040372.xml")
    
