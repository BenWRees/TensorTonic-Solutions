import numpy as np

def bag_of_words_vector(tokens: list, vocab: list) -> np.ndarray:
    """
    Returns a NumPy array with length len(vocab).
        Want to count how often each element in the vocab list occurs in the tokens list
    """
    count_tokens = dict() 
    for token in tokens :
        if token in count_tokens :
            count_tokens[token] += 1
        else :
            count_tokens[token] = 1

    bag_of_words_count = []
    for token in vocab : 
        if token not in count_tokens : 
            bag_of_words_count.append(0)
        else : 
            bag_of_words_count.append(count_tokens[token])

    return np.array(bag_of_words_count)
    