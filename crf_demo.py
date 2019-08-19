# Driver Function & commandline options for CRF
# serves as a demo for CRF NER
# based on boilerplate from https://www.tutorialspoint.com/python/python_command_line_arguments.htm
# modified by Santi(chai) Pornavalai
# 31.3.2019
# tested on python 3.7.2

import sys, getopt
#from crf import CRF
from crf import CRF
from vectorizer import Vectorizer

def main(argv):
 
    inputpath = ''
    testpath = ''
    outputfile = ''
    iterations = 5

    evaluate = False
    save_to_bin = False

    idx_fname = ""
    weights_fname = ""

    try:
        opts, args = getopt.getopt(argv,"hei:o:t:k:s",["ifile=","ofile=","iter=","tfile=","eval", \
                                        "save","load-weights=","load-index="])
    except getopt.GetoptError:
        print ("""ERROR!! General USAGE: crf_demo.py -i <PATH TO TRAIN FILE> -t <TEST FILE NAME> -o <RESULT FILE NAME>
                   for other options rtm """)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('crf_demo.py -i <PATH TO TRAIN FILE> -o <RESULT FILE NAME>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputpath = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-k", "--iter"):
            iterations = arg
        elif opt in ("-t","--tfile"):
            testpath = arg
        elif opt in ("e,--eval"):
            evaluate = True
        elif opt in ("--save","s"):
            save_to_bin = True
        elif opt in ("--load-index",):
            idx_fname = arg
        elif opt in ("--load-weights",):
            weights_fname = arg

        
    crf_model = CRF()
    if len(inputpath) == 0:

        if len(weights_fname) > 0 and len(idx_fname) > 0:
            crf_model.load_weights(weights_fname,idx_fname)
        else:
            print("No weight or index file provided")
            sys.exit(3)
    else:
        crf_model.fit(inputpath, iterations=10)

    if len(testpath) > 0:
        predicted, real = crf_model.predict(testpath)

        if evaluate:
            from sklearn.metrics import classification_report as report
            tags = crf_model.vectorizer.tag_list
            print("f1 score:" ,report(real,predicted,labels=tags))

        with open(outputfile, "w") as testf:
            for pred,reel in zip(predicted,real):
                testf.write(pred +"\t" + reel + "\n" )

    if save_to_bin:
        wfile_name = outputfile + ".weights"
        indfile_name = outputfile + ".index"
        crf_model.save_weights(wfile_name, indfile_name)


if __name__ == "__main__":
    main(sys.argv[1:])