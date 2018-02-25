tags = {
    'CC':   0,
    'CD':   1,
    'DT':   2,
    'EX':   3,
    'FW':   4,
    'IN':   5,
    'JJ':   6,
    'JJR':  7,
    'JJS':  8,
    'LS':   9,
    'MD':   10,
    'NN':   11,
    'NNS':  12,
    'NNP':  13,
    'NNPS': 14,
    'PDT':  15,
    'POS':  16,
    'PRP':  17,
    'PRP$': 18,
    'RB':   19,
    'RBR':  20,
    'RBS':  21,
    'RP':   22,
    'TO':   23,
    'UH':   24,
    'VB':   25,
    'VBD':  26,
    'VBG':  27,
    'VBN':  28,
    'VBP':  29,
    'VBZ':  30,
    'WDT':  31,
    'WP':   32,
    'WP$':  33,
    'WRB':  34}


def tag_to_vec(tag):
    vec = [0] * len(tags)

    try:
        vec[tags[tag]] = 1
    except KeyError as e:
        return None

    return vec
