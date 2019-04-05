import os
import re
import string
import numpy as np

dataDir = '/u/cs401/A3/data/'
# dataDir = '../data/'


def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    # >>> wer("who is there".split(), "is there".split())
    0.333 0 0 1                                                                           
    # >>> wer("who is there".split(), "".split())
    1.0 0 0 3                                                                           
    # >>> wer("".split(), "who is there".split())
    Inf 0 3 0                                                                           
    """

    n = len(r)
    m = len(h)
    R = np.zeros((n + 1, m + 1))
    B = np.zeros((n + 1, m + 1))

    for i in range(n + 1):
        for j in range(m + 1):
            if i is 0 or j is 0:
                R[i, j] = i if j is 0 else j
                B[i, j] = 1 if j is 0 else 2

    for i in range(n):
        i = i + 1
        for j in range(m):
            j = j + 1

            t_del = R[i - 1, j] + 1
            t_sub = R[i - 1, j - 1] + (0 if r[i - 1] == h[j - 1] else 1)
            t_ins = R[i, j - 1] + 1

            R[i, j] = min(t_del, min(t_sub, t_ins))

            if R[i, j] == t_del:
                B[i, j] = 1
            elif R[i, j] == t_ins:
                B[i, j] = 2
            else:
                B[i, j] = 3

    subs = 0
    ins = 0
    dels = 0

    i = n
    j = m

    while i >= 0 and j >= 0 and i + j > 0:

        if B[i, j] == 1:
            i -= 1
            dels += 1
        elif B[i, j] == 2:
            j -= 1
            ins += 1
        elif B[i, j] == 3:
            i -= 1
            j -= 1
            if r[i] != h[j]:
                subs += 1

    wer = (subs + ins + dels) / n if n != 0 else float('inf')

    return wer, subs, ins, dels


def preProcess(line):
    line = line.translate(str.maketrans('', '', re.sub(r"[\[|\]]", "", string.punctuation))).lower().split()

    return line


if __name__ == "__main__":

    f = open("asrDiscussion.txt", 'w+')

    google_wer = []
    kaldi_wer = []

    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            file = open(dataDir + speaker + '/transcripts.txt', "r")
            file_google = open(dataDir + speaker + '/transcripts.Google.txt', "r")
            file_kaldi = open(dataDir + speaker + '/transcripts.Kaldi.txt', "r")

            lines = []
            reference = []
            for line in file:
                lines.append(line)
                reference.append(preProcess(line))

            google = []
            for line in file_google:
                google.append(preProcess(line))

            kaldi = []
            for line in file_kaldi:
                kaldi.append(preProcess(line))

            for i in range(len(reference)):
                wer, subs, ins, dels = Levenshtein(reference[i], google[i])
                google_wer.append(wer)
                f.write(f"[{speaker}] [Google] [{i}] [{wer}] S:[{subs}], I:[{ins}], D:[{dels}] \n")

                wer, subs, ins, dels = Levenshtein(reference[i], kaldi[i])
                kaldi_wer.append(wer)
                f.write(f"[{speaker}] [Kaldi] [{i}] [{wer}] S:[{subs}], I:[{ins}], D:[{dels}] \n")

            file.close()
            file_google.close()
            file_kaldi.close()

    f.write(f"Google wer average is {np.average(google_wer)}, standard deviation is {np.std(google_wer)}. \n")
    f.write(f"Kaldi wer average is {np.average(kaldi_wer)}, standard deviation is {np.std(kaldi_wer)}. \n")

    f.write("The Substitution errors are playing important roles in both Google and Kaldi results."
            " That implies that both tools are correctly getting most words."
            " In the meanwhile, Kaldi give better result in the overall testing")

    f.close()
