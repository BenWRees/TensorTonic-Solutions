def word_count_dict(sentences: list) -> dict:
    """
    Returns a dictionary of token counts.
    """
    # Write code here
    sentences_flatten = [a for b in sentences for a in b]
    count_dict = dict() 

    for token in sentences_flatten :
        if token not in count_dict : 
            count_dict[token] = 1 
        else : 
            count_dict[token] += 1

    return count_dict