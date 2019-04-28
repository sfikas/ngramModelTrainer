import unittest
import sys
import fileinput
import numpy as np
import scipy.io
import tqdm
import itertools
import argparse
import logging
from collections import Counter
from os.path import basename

def strip(word):
    # Accepts a string as input, 
    # returns a string where only characters in alphabet remain.
    res = []
    word = word.strip()
    #for c in word.lower():
    for c in word:
        if c in alphabet:
            res += c
        else:
            raise Warning('Character {} not in alphabet!'.format(c))
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

def countQuadgrams(word):
    res = Counter()
    l = list(word)
    for i in range(0, len(word)-3):
        res[(l[i], l[i+1], l[i+2], l[i+3])] += 1
    return res

def computeQuadgramsJointPdf(quadgrams):
    # This is the (joint) probability of tetragrams ("quadgrams")
    res = np.zeros( (len(alphabet), len(alphabet), len(alphabet), len(alphabet)) )
    for i in range(0, len(alphabet)):
        for j in range(0, len(alphabet)):
            for k in range(0, len(alphabet)):
                for l in range(0, len(alphabet)):
                    res[i, j, k, l] = quadgrams[alphabet[i], alphabet[j], alphabet[k], alphabet[l]]
                    if res[i, j, k, l] == 0:
                        res[i, j, k, l] = np.finfo(float).eps                
    res = res / res.flatten().sum()
    return res

def computeQuadgramsConditionalPdf(quadgramsJointPdf, trigramsJointPdf):
    res = np.zeros( (len(alphabet), len(alphabet), len(alphabet), len(alphabet)) )
    for previous in range(0, len(alphabet)):
        for anteprevious in range(0, len(alphabet)):
            for paranteprevious in range(0, len(alphabet)):
                for i in range(0, len(alphabet)):
                    res[paranteprevious, anteprevious, previous, i] = quadgramsJointPdf[paranteprevious, anteprevious, previous, i] / trigramsJointPdf[paranteprevious, anteprevious, previous]
    #Renormalize (to avoid numerical errors)
    for previous in range(0, len(alphabet)):
        for anteprevious in range(0, len(alphabet)):
            for paranteprevious in range(0, len(alphabet)):
                res[paranteprevious, anteprevious, previous, :] = res[paranteprevious, anteprevious, previous, :] / res[paranteprevious, anteprevious, previous, :].flatten().sum()
    return res

def main(filename, count_tetragrams=False):
    unigrams = Counter()
    bigrams = Counter()
    trigrams = Counter()
    if count_tetragrams:
        print('Computing tetragrams. This will normally take *a lot* of time..')
        quadgrams = Counter()
    fin = fileinput.input(filename)
    for word in tqdm.tqdm(fin):
        strippedword = strip(word)
        unigrams += countUnigrams(strippedword)
        bigrams += countBigrams(strippedword)
        trigrams += countTrigrams(strippedword)
        if count_tetragrams:
            quadgrams += countQuadgrams(strippedword)
    unigramsPdf = computeUnigramsPdf(unigrams)
    bigramsJointPdf = computeBigramsJointPdf(bigrams)
    trigramsJointPdf = computeTrigramsJointPdf(trigrams)
    if count_tetragrams:
        quadgramsJointPdf = computeQuadgramsJointPdf(quadgrams)
    else:
        quadgramsJointPdf = None
    return unigramsPdf, bigramsJointPdf, trigramsJointPdf, quadgramsJointPdf

class TestMethods(unittest.TestCase):
    def test_unigrams(self):
        unigramsPdf, b, c, _ = main('fixtures/dummy.txt')
        self.assertTrue(abs(unigramsPdf[alphabet.index('a')] - 5./35.) < 1e-5)
        self.assertTrue(abs(unigramsPdf[alphabet.index('z')]) < 1e-5)

    def test_bigrams(self):
        unigramsPdf, bigramsJointPdf, c, _ = main('fixtures/dummy.txt')
        bigramsPdf = computeBigramsConditionalPdf(bigramsJointPdf, unigramsPdf)
        self.assertTrue(abs(bigramsPdf[alphabet.index('l'), alphabet.index('l')] - 1./4.) < 1e-5)

    def test_trigrams(self):
        unigramsPdf, bigramsJointPdf, trigramsJointPdf, _ = main('fixtures/dummy.txt')
        trigramsPdf = computeTrigramsConditionalPdf(trigramsJointPdf, bigramsJointPdf)
        self.assertTrue(abs(trigramsPdf[alphabet.index('a'), alphabet.index('i'), alphabet.index('n')] - 1.) < 1e-5)
        self.assertTrue(abs(trigramsPdf[alphabet.index('t'), alphabet.index('h'), alphabet.index('e')] - 1.) < 1e-5)
        self.assertTrue(abs(trigramsPdf[alphabet.index('x'), alphabet.index('y'), alphabet.index('z')]) < 2./36.)

    def test_quadgrams(self):
        unigramsPdf, bigramsJointPdf, trigramsJointPdf, quadgramsJointPdf = main('fixtures/dummy2.txt')
        quadgramsPdf = computeQuadgramsConditionalPdf(quadgramsJointPdf, trigramsJointPdf)
        self.assertTrue(abs(quadgramsPdf[alphabet.index('r'), alphabet.index('s'), alphabet.index('o'), alphabet.index('n')] - 1.) < 1e-5)

if __name__ == "__main__":
    logger = logging.getLogger('::ngramModelTrainer::')
    logger.info('-------------------')
    parser = argparse.ArgumentParser()
    parser.add_argument('--alphabet', required=False, choices=['almazan36', 'dummy', 'dutta_extended', 'sophia'], default='almazan36', help='Alphabet to be used.')
    parser.add_argument('--input_corpus', dest='input_corpus', default='', help='Input corpus. Each line must be a separate word. If none is given, a simple test is run.')
    parser.add_argument('--compute_tetragrams', dest='compute_tetragrams', action='store_true', help='Compute tetragrams. Default: False (normally too expensive, unless a big corpus with relatively few unigrams is available)')
    parser.set_defaults(
            alphabet='almazan36', 
            input_corpus='',
            compute_tetragrams=False
        )
    args = parser.parse_args()

    if args.alphabet == 'almazan36':
        alphabet = [
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
            'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
            'u', 'v', 'w', 'x', 'y', 'z',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',    
        ]
    elif args.alphabet == 'dummy':
        alphabet = [
            'a', 'b', 'c',
        ]
    elif args.alphabet == 'dutta_extended':
        alphabet = [chr(i) for i in itertools.chain(
                    range(ord('a'), ord('z') + 1),
                    range(ord('A'), ord('Z') + 1),
                    range(ord('0'), ord('9') + 1),
                )
            ] + ['.', ',', '[', '-', ';', '"', '/', ':', '=', '(', ')', '>', '+', '!', '#', '@', '~', '#']
    elif args.alphabet == 'sophia':
        alphabet = ['ὸ', 'ὴ', 'ὕ', 'ὄ', 'έ', 'ι', '4', 'ἄ', 'ά', 'ὼ', 'ἔ', 'ᾶ', 'ἴ', 'ὃ', '6', '3', 'ή', 'ἰ', 'ὥ', 'φ', 
        'ῥ', 'λ', 'ᾳ', 'ξ', 'χ', 'ἶ', 'ῄ', 'ὔ', 'ἑ', 'ψ', 'ύ', 'θ', 'ώ', 'ἣ', 'ϊ', '0', 'τ', 'ἧ', 'ν', 'ἤ', 'ό', 'ἡ', 'β', 'σ', 'ἃ', 
        'ῳ', 'ἀ', '8', 'ῆ', 'ῖ', 'ω', '5', 'ῷ', 'ἥ', 'ἂ', 'γ', '2', 'ῶ', 'ΐ', '1', 'μ', 'ἠ', '9', 'ρ', 'ὰ', 'ἢ', 'ὺ', 'ἷ', 'ἁ', 'ἅ', 
        'π', 'ὶ', 'ἕ', 'ἵ', 'δ', 'ἐ', 'ὠ', '#', 'ἱ', 'ὑ', 'κ', 'ὀ', 'ζ', 'ὲ', 'ὐ', 'ο', 'ί', 'ῇ', 'ε', 'é', '7', 'α', 'ἦ', 'ς', 'ῃ', 
        'ῦ', 'υ', 'ὗ', 'ὅ', 'ὡ', 'η', 'ὁ', 'ἓ']
        '''
        alphabet = ['ὑ', 'α', '6', 'Ἑ', 'ὥ', 'Ἱ', 'μ', 'L', 'y', 'δ', 'ῦ', 'Β', 'Α', 'ὺ', 'O', 'ἥ', 'S', 'ρ', 'd', 'w', 'Ἀ', 'B', 
        'ὗ', 'Ὁ', 'Ὑ', 'Ν', 'ᾳ', '8', 'Σ', 'M', 'A', 'Ἦ', 'V', 'κ', '2', 'ῖ', 'g', 'T', 'ἰ', '#', 'r', 'ὐ', 'ἤ', 'β', 'c', 'ῄ', 'ἷ', 'Μ', 'ὄ', 
        '5', 'ἕ', 'W', 'ϊ', 'Κ', 'ῷ', 'Χ', 'ὁ', 'm', 'ὕ', 'P', 'Ἐ', 'f', 'ὼ', 'υ', 'ν', 'D', 'ἑ', 'ή', 'Ἕ', 'ἀ', 'Ἔ', 'b', 'ώ', 'ὀ', 'ὴ', 'é', 'I', 
        'Ρ', 'Ὄ', 'Ἠ', 'ἧ', 'ό', 'Ο', '0', 's', '1', 'o', 'K', 'ἁ', 'ἠ', 'ἶ', 'u', 'ἣ', 'ξ', 'λ', 'ι', 'ἦ', 'a', 'Ε', 'ἴ', 'γ', 'k', 'ί', 'ὰ', 'G', 
        'Ἡ', 'η', 'Θ', 'Τ', 'ἂ', 'C', 'ς', 'Ἰ', 'ὃ', 'ὲ', 'v', '3', 'Φ', 'φ', 'ἄ', 'ύ', 'ἵ', 'p', 'θ', 'ῆ', 'F', 'ζ', 'ά', 'i', 'x', 'n', 'τ', 'π', 
        'ὡ', 'Λ', 'Δ', 'ῇ', 'ὅ', 'l', 'έ', 'ῳ', 'ε', 'ὸ', 'E', 'R', 'Ἁ', 'ὔ', 'Ἅ', 'ω', '4', 'h', 'ΐ', 'ῥ', 'z', 'ἢ', 'ἔ', 'Ἄ', '9', 'ἅ', 'ο', 'σ', 
        'ῶ', 'ἐ', 'Γ', 'e', 'ὶ', '7', 'ἡ', 'ἓ', 'Ἂ', 'Ὕ', 't', 'ῃ', 'ἱ', 'Π', 'ὠ', 'Ὅ', 'ἃ', 'χ', 'ᾶ', 'ψ']
        '''
    else:
        raise NotImplementedError('Unknown alphabet.')

    if args.input_corpus != '':
        filename = args.input_corpus
        unigramsPdf, bigramsJointPdf, trigramsJointPdf, quadgramsJointPdf = main(filename, count_tetragrams=args.compute_tetragrams)
        #print(np.log(computeBigramsConditionalPdf(bigramsJointPdf, unigramsPdf)))
        #print(np.log(computeTrigramsConditionalPdf(trigramsJointPdf, bigramsJointPdf)))
        savedict = dict(
            alphabet=alphabet,
            unigrams=unigramsPdf, 
            bigrams=computeBigramsConditionalPdf(bigramsJointPdf, unigramsPdf),
            trigrams=computeTrigramsConditionalPdf(trigramsJointPdf, bigramsJointPdf),
            )
        if args.compute_tetragrams:
            savedict['quadgrams'] = computeQuadgramsConditionalPdf(quadgramsJointPdf, trigramsJointPdf)
        scipy.io.savemat(basename(filename)+'.ngrams.mat', savedict)
    else:
        unittest.main()