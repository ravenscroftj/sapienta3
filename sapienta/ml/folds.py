"""
Handle training fold information stored in CSV file
"""

import logging
import csv
import sys

logger = logging.getLogger(__name__)

def get_folds( inputfile ):
    """Get fold information from the given input CSV file"""

    with open(inputfile, 'rb') as f:
        foldreader = csv.reader(f, delimiter=',', quotechar='"')

        #read the first row to detect how many folds there are
        toprow = foldreader.next()

        foldcount = len([x for x in toprow if x != ""])

        logger.info("%d folds have been detected",foldcount)
        
        #read the next row and make sure the number of cells is divisable by folds
        labels = foldreader.next()[1:]

        if len(labels) % foldcount != 0:
            logger.error("The number of labels doesn't factor with the number of folds")
            return None

        cols = len(labels) / foldcount

        logger.info("Detected %d information columns per fold", cols)

        colnames = [ labels[x] for x in range(0,cols)]

        extracted = []

        #now read the rest of the rows
        for row in foldreader:
            #we discount row[0] because that's just a file number
            foldinfos = row[1:]
            extracted.append( extract_fold_entries( foldinfos, foldcount, colnames ) )

    #return the folds
    return [ [ex[i] for ex in extracted]  for i in range(0, foldcount)]

def extract_fold_entries( row, foldc, colnames):
    """Given a single row from the CSV, extract all for all folds"""
    extracted = []

    for i in range(0, foldc):
            extracted.append( { colnames[x] : row[x+(len(colnames)*i)]  for x in range(0,len(colnames))} )

    return extracted


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    folds = get_folds("/home/james/tmp/foldTable.csv")

    for i in range(0,len(folds)):
        print ("Fold #%d: %s" % ( i+1, str(folds[i])))
