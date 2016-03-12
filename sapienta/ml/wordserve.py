"""
Wordserv - simple server application that provides mapping service to/from word vectors over HTTP

"""
import sys
import requests
import numpy as np

from flask import jsonify, Flask, request

from gensim.models.word2vec import Word2Vec

app = Flask("wordserv")

_wv = None

help =  """

<html><head><title>WordServ</title></head>

<body>

<h1>WordServ</h1>
<h2>A simple word2vec server By <a href="http://brainsteam.co.uk">James Ravenscroft</a></h2>

<h3>POST /vector</h3>
<p>Make a post request to /vector endpoint with a form variable 'word' to receive a wordvector for that word 
or post an array of words and get an array of words back.
</p>

<h4>Example</h4>
<pre>
curl -X POST -F word=hello -F word=goodbye http://localhost:5000/vector 
</pre>

</body>
</html>

"""


class WordservClient:
    
    def __init__(self, host, port):
        self.root = "http://{}:{}".format(host,port)
        
    def vector(self, words):
        r = requests.post(self.root+"/vector", data={"word" : words}).json()
        return { w : np.array(r['wordvectors'][w]) for w in r['wordvectors'] }


@app.route("/")
def index():
    return help

@app.route("/vector", methods=['POST'] )
def vector():
    global _wv
    
    wordvecs = {}
    
    for word in request.form.getlist('word'):
        if word in _wv:
            wordvecs[word] = _wv[word].tolist()
        else:
            wordvecs[word] = [0] * 300
        
    return jsonify(wordvectors=wordvecs)
    

def main():
    global _wv
    
    import argparse
    
    ap = argparse.ArgumentParser()
    
    ap.add_argument("vector_file", action="store", help="Path to the word2vec file to serve")
    ap.add_argument("-t", "--testclient", dest="test", action="store_true", help="If set, runs the test client against an existing server instance")
    ap.add_argument("--host", dest="host", action="store", default="localhost", help="The host to bind to, defaults to localhost")
    ap.add_argument("-p", "--port", dest="port", action="store", help="The port to serve on, defaults to 5000", default=5000)
    ap.add_argument("-d", "--debug", dest="debug", action="store_true", help="If true, provides debug output")
    
    args = ap.parse_args()
    
    if args.test:
        print("Running test client against http://{}:{}".format(args.host,args.port))
        
        tc = WordservClient(args.host,args.port)
        
        for i in range(0,500):
            vecs = tc.vector(["hello","world"])
        
        sys.exit(1)
    
    print("Loading word vector - this might take a while...")
    
    _wv = Word2Vec.load_word2vec_format("/opt/GoogleNews-vectors-negative300.bin.gz", binary=True)
    
    app.run(port=args.port, debug=args.debug)
    
if __name__ == "__main__":
    main()