# Named Entity Recognition based on Conditional Random Fields

Python module for Linear Conditional Random Fields applied to Named Entity Recognition
Santichai Pornavalai
31.3.19

## Setup:
run 
*pip3 install -r requirements.txt*
## Usage:


python crf_demo.py 

-i <PATH TO INPUT FILE>  in GermEval BIO format. Ignores # and empty lines etc.
-o <NAME OF RESULT FILE> a tsv file. <TOKEN>    <PREDICTED>     <REAL>
-t <NAME OF TEST FILE>    Test file in GermEVAL format
-k  <NUMBER>  number of iterations. Default is 5
--eval          prints classifications report detailing F1-score, accuracy, precision, recall for all tags. requires Scikit-learn

--load-index <INDEX-BINARY>     loads vectorizer from binary
--load-weights <WEIGHT-BINARY>  loads weights from binary

-s      save to binary. Creates two binary files in working directory.  

### example usage:

1.
*python crf_demo.py -i data/train -t data/test -o results.txt -k 10 --eval --save*

train on training data 10 times. test on test data. write results to results.txt. print report. save to binary

2.

*python crf_demo.py  -t data/minimini -o results.txt --eval --load-index saved_bin.index --load-weights saved_bin.weights*

loads from weights, tests on test data, write results to results.txt, print classification report



## Intro

This module reads a tab separated training file in BIO notation. The model was developed, tested and trained on data from GermEval 2014 shared task downloaded
directly from their homepage. Two main classes are provided, Vectorizer and CRF. The CRF class makes use of but does not inherit from Vectorizer. 

The CRF class handles training and inference has two main objects, the weights and vectorizer. Training is done using the averaged perceptron algorithm, whereas
decoding employs a variant of the Viterbi algorithm. The main API functions are fit and predict, which in turn wraps numerous initializations and functions together. 
to add more features edit the fit function. 

The Vectorizer class is supposed to handle feature extraction etc. but has since grown to become a catch-all class for a bunch of utility functions. The fit method builds indices
for the words and tags as well as keeping track of the feature vector. The transform method is used to transform token sequences in to sparse vectors. Feature extractors are also methods
in this class. I have only provided a few features, since feature engineering is an art on its own right. It is however easy to implement new features. Basically any function can be 
added to the features by calling add_feature() on a function object. This function object needs to take a few arguments such as token sequence and position etc. and return the position 
in which it is be added in the feature vector. 

## Evaluation

To verify the correctness of the training algorithm, I first set out to see if the CRF can basically overfit on its own data. This shows that at least it is able to learn. 

Next I ran the CRF inference on unseen data.  Although the accuracy was high, it seemed to be a bad evaluation metric since NER data is generally extremely imbalanced with most of the tags being OTHER. F1 and precision/recall for each individual tag were used
instead. 

The frequent tags seemed to get around 0.55 to 0.6 F score across the board. This show's that atleast it is better that random. Bare in mind that this CRF
is basically just a HMM with a gazeteer and some other minimal features. Precision seemed to be always be higher than recall. 

The next logical step would be to implement more feature functions and do a series of cross validations etc. I didn't do this...

## Implementation Details

The goal of this project was to implement a CRF model specifically but not limited to NER from more or less scratch. 
At first the idea was to use a numeric library such as numpy to do most of the intensive calculations. Earlier verions using
dense Numpy arrays yielded correct results on toy data sets. This was however problematic since CRFs typically require very large vectors. The next step was to use the sparse array
implementation from Scipy (Numpy's sparse cousin so to say). I soon discovered that they required quite a large amount of overhead when initializing small sparse arrays. This makes it slightly less
suitable for CRFs which need to "try out" different states while decoding. In the end I settled on using vanilla python dictionaries. The reasoning for this is two-fold: firstly, it does not
require any large dependencies or other external sources. Secondly, it is a well implmented hash map which can be used to implement the "Hashing Trick". As a sanity check, I tested
scalar product operations on several sparse arrays using different data structures. At below 5% density on an array with around 1 million elements , dictionaries outperformed numpy significantly. 
Scipy sparse matrices seemed to suffer initialized inside a for loop.

## Optimizations
- only consider known transitions in Viterbi
- precalculate feature vectors for gold data in the first iteration
- Viterbi has an option to output a list of feature vectors. This avoids unnecessary recalculation during perceptron training
- Only pickout feature vectors from words that don't match during perceptron training. 

