import sys
import fileinput
import numpy as np
from collections import Counter

alphabet = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',    
]

def strip(word):
    # Accepts a string as input, 
    # returns a string where only characters in alphabet remain.
    res = []
    for c in word.lower():
        if c in alphabet:
            res += c
    return ''.join(res)

def countUnigrams(word):
    res = Counter()
    l = list(word)
    for i in range(0, len(word)):
        res[l[i]] += 1
    return res

def computeUnigramsPdf(unigrams):
    res = np.zeros(len(alphabet))
    for i in range(0, len(alphabet)):        
        res[i] = unigrams[alphabet[i]]
        if res[i] == 0:
            res[i] = np.finfo(float).eps
    res = res / res.sum()
    return res

def countBigrams(word):
    res = Counter()
    l = list(word)
    for i in range(0, len(word)-1):
        res[(l[i], l[i+1])] += 1
    return res

def computeBigramsJointPdf(bigrams):
    # This is the (joint) probability of bigrams
    res = np.zeros( (len(alphabet), len(alphabet)) )
    for i in range(0, len(alphabet)):
        for j in range(0, len(alphabet)):
            res[i, j] = bigrams[alphabet[i], alphabet[j]]
            if res[i, j] == 0:
                res[i, j] = np.finfo(float).eps
    res = res / res.flatten().sum()
    return res

def computeBigramsConditionalPdf(bigramsJointPdf, unigramsPdf):
    # This is the conditional probability of bigrams
    # The argument 'previous' is an index pointing to the alphabet letter that is
    # assumed to be the pdf condition.
    # Formally: p( . | previous )

    # jointpdf will be p( x, x-1 = previous )
    # which breaks down as p( x | x-1 = previous ) p( x-1 = previous ),
    # hence dividing by p ( x-1 = previous ) leaves us with the conditional probability.
    res = np.zeros((len(alphabet), len(alphabet)))
    for previous in range(0, len(alphabet)):
        for i in range(0, len(alphabet)):
            res[previous, i] = bigramsJointPdf[previous, i] / unigramsPdf[previous]
    #Renormalize (to avoid numerical errors)
    for previous in range(0, len(alphabet)):
        res[previous, :] = res[previous, :] / res[previous, :].flatten().sum()
    return res

def countTrigrams(word):
    res = Counter()
    l = list(word)
    for i in range(0, len(word)-2):
        res[(l[i], l[i+1], l[i+2])] += 1
    return res

def computeTrigramsJointPdf(trigrams):
    # This is the (joint) probability of trigrams
    res = np.zeros( (len(alphabet), len(alphabet), len(alphabet)) )
    for i in range(0, len(alphabet)):
        for j in range(0, len(alphabet)):
            for k in range(0, len(alphabet)):
                res[i, j, k] = trigrams[alphabet[i], alphabet[j], alphabet[k]]
                if res[i, j, k] == 0:
                    res[i, j, k] = np.finfo(float).eps                
    res = res / res.flatten().sum()
    return res

def computeTrigramsConditionalPdf(trigramsJointPdf, bigramsJointPdf):
    res = np.zeros( (len(alphabet), len(alphabet), len(alphabet)) )
    for previous in range(0, len(alphabet)):
        for anteprevious in range(0, len(alphabet)):
            for i in range(0, len(alphabet)):
                res[anteprevious, previous, i] = trigramsJointPdf[anteprevious, previous, i] / bigramsJointPdf[anteprevious, previous]
    #Renormalize (to avoid numerical errors)
    for previous in range(0, len(alphabet)):
        for anteprevious in range(0, len(alphabet)):
            res[anteprevious, previous, :] = res[anteprevious, previous, :] / res[anteprevious, previous, :].flatten().sum()
    return res


if __name__ == "__main__":
    unigrams = Counter()
    bigrams = Counter()
    trigrams = Counter()
    for word in fileinput.input([sys.argv[1]]):
        strippedword = strip(word)
        unigrams += countUnigrams(strippedword)
        bigrams += countBigrams(strippedword)
        trigrams += countTrigrams(strippedword)
    unigramsPdf = computeUnigramsPdf(unigrams)
    bigramsJointPdf = computeBigramsJointPdf(bigrams)
    trigramsJointPdf = computeTrigramsJointPdf(trigrams)

    print(np.log(computeBigramsConditionalPdf(bigramsJointPdf, unigramsPdf)))
    print(np.log(computeTrigramsConditionalPdf(trigramsJointPdf, bigramsJointPdf)))
    #print(bigramsPdf)
    #print(trigramsPdf)