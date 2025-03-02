import sys
import math


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename): # was going to do this like leetcode #49, but need to return a hashmap according to 1.1
    # get count of each char
    count = {}
    for i in range(65, 91): # use ASCII range
        count[chr(i)] = 0 # initialize to 0
        
    with open (filename,encoding='utf-8') as f: # open our file
        for char in f.read(): # read char by char
            if 'A' <= char.upper() <= 'Z':  # if it's alpha, increment count
                count[char.upper()] += 1
    return count 

def main():
    if len(sys.argv) != 4: 
        print("Error! Make sure to have argv of length 4!")
        return # just return here, no need to attempt calculation
    
    X = shred(sys.argv[1]) # get our map from shread fn we created
    
    # now we'll set our variables for Bayes Rule
    e, s = get_parameter_vectors()
    
    # this comes straight from comamand line arguments (our 'priors')
    p_e, p_s = float(sys.argv[2]), float(sys.argv[3]) # P(Y = English) = p_e, P(Y = Spanish) = p_s
    
    # log(P(X | Y = y) * P(Y = y)) = log(prior) + sum from 1 to 26 of (Xi * log(pi))
    # take log of both sides to simplify -- C(X) cancels out
    log_P_english = math.log(p_e)
    for i in range(26):
        log_P_english += X[chr(i + ord('A'))] * math.log(e[i])

    log_P_spanish = math.log(p_s)
    for i in range(26):
        log_P_spanish += X[chr(i + ord('A'))] * math.log(s[i])
    
    # by here, we've calculated the numerator of the equation (log_P_english | log_P_spanish), and now we just need the denominator
    
    # compute posterior probability of English
    likelihood_gap = log_P_spanish - log_P_english
    if -100 < likelihood_gap < 100:
        p_english = 1 / (1 + math.exp(likelihood_gap))
    elif likelihood_gap >= 100:
        p_english = 0
    else:
        p_english = 1.0
    
    print("Q1")
    for char in sorted(X.keys()): # sort first so alphabetical (even though it should already be sorted by the way we instantiated it)
        print(f"{char} {X[char]}")
    
    print("Q2")
    print(f"{(X['A'] * math.log(e[0])):.4f}")
    print(f"{(X['A'] * math.log(s[0])):.4f}")
    
    print("Q3")
    print(f"{log_P_english:.4f}")
    print(f"{log_P_spanish:.4f}")
    
    print("Q4")
    print(f"{p_english:.4f}")

if __name__ == "__main__":
    main()

    
    