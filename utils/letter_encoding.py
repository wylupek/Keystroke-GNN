# know letters 
# define encoding for each letter
from enum import Enum

def iota():
    x = 0
    while True:
        yield x
        x += 1

iota_gen = iota()

class Alphabet(Enum):
    A = next(iota_gen)
    Ą = next(iota_gen)
    B = next(iota_gen)
    C = next(iota_gen)
    Ć = next(iota_gen)
    D = next(iota_gen)
    E = next(iota_gen)
    Ę = next(iota_gen)
    F = next(iota_gen)
    G = next(iota_gen)
    H = next(iota_gen)
    I = next(iota_gen)
    J = next(iota_gen)
    K = next(iota_gen)
    L = next(iota_gen)
    Ł = next(iota_gen)
    M = next(iota_gen)
    N = next(iota_gen)
    Ń = next(iota_gen)
    O = next(iota_gen)
    Ó = next(iota_gen)
    P = next(iota_gen)
    Q = next(iota_gen)
    R = next(iota_gen)
    S = next(iota_gen)
    Ś = next(iota_gen)
    T = next(iota_gen)
    U = next(iota_gen)
    W = next(iota_gen)
    V = next(iota_gen) 
    Y = next(iota_gen)
    Z = next(iota_gen)
    Ź = next(iota_gen)
    Ż = next(iota_gen)
    X = next(iota_gen)

    COMMA = next(iota_gen)
    DOT = next(iota_gen)
    EXCLAMATION_MARK = next(iota_gen)
    SPACE = next(iota_gen)
    QUESTION_MARK = next(iota_gen)
    NEWLINE = next(iota_gen)
    HASH = next(iota_gen)
    STAR = next(iota_gen)
    DEL = next(iota_gen)
    AT = next(iota_gen)
    COLON = next(iota_gen)
    SEMICOLON = next(iota_gen)
    QUOTE = next(iota_gen)
    DOUBLE_QUOTE = next(iota_gen)
    LEFT_PAREN = next(iota_gen)
    RIGHT_PAREN = next(iota_gen)
    LEFT_BRACKET = next(iota_gen)
    RIGHT_BRACKET = next(iota_gen)
    LEFT_BRACE = next(iota_gen)
    RIGHT_BRACE = next(iota_gen)
    LESS_THAN = next(iota_gen)
    GREATER_THAN = next(iota_gen)
    SLASH = next(iota_gen)
    BACKSLASH = next(iota_gen)
    PIPE = next(iota_gen)
    AMPERSAND = next(iota_gen)
    PERCENT = next(iota_gen)
    DOLLAR = next(iota_gen)
    CARET = next(iota_gen)
    TILDE = next(iota_gen)
    UNDERSCORE = next(iota_gen)
    PLUS = next(iota_gen)
    MINUS = next(iota_gen)
    EQUAL = next(iota_gen)

    a = next(iota_gen)
    ą = next(iota_gen)
    b = next(iota_gen)
    c = next(iota_gen)
    ć = next(iota_gen)
    d = next(iota_gen)
    e = next(iota_gen)
    ę = next(iota_gen)
    f = next(iota_gen)
    g = next(iota_gen)
    h = next(iota_gen)
    i = next(iota_gen)
    j = next(iota_gen)
    k = next(iota_gen)
    l = next(iota_gen)
    ł = next(iota_gen)
    m = next(iota_gen)
    n = next(iota_gen)
    ń = next(iota_gen)
    o = next(iota_gen)
    ó = next(iota_gen)
    p = next(iota_gen)
    q = next(iota_gen)
    r = next(iota_gen)
    s = next(iota_gen)
    ś = next(iota_gen)
    t = next(iota_gen)
    u = next(iota_gen)
    w = next(iota_gen)
    v = next(iota_gen)
    y = next(iota_gen)
    z = next(iota_gen)
    ź = next(iota_gen)
    ż = next(iota_gen)
    x = next(iota_gen)
    ZERO = next(iota_gen)
    ONE = next(iota_gen)
    TWO = next(iota_gen)
    THREE = next(iota_gen)
    FOUR = next(iota_gen)
    FIVE = next(iota_gen)
    SIX = next(iota_gen)
    SEVEN = next(iota_gen)
    EIGHT = next(iota_gen)
    NINE = next(iota_gen)

    
AlphabetSize = len(Alphabet)

def char_to_enum(char):
    char_map = {
        'A': Alphabet.A,
        'Ą': Alphabet.Ą,
        'B': Alphabet.B,
        'C': Alphabet.C,
        'Ć': Alphabet.Ć,
        'D': Alphabet.D,
        'E': Alphabet.E,
        'Ę': Alphabet.Ę,
        'F': Alphabet.F,
        'G': Alphabet.G,
        'H': Alphabet.H,
        'I': Alphabet.I,
        'J': Alphabet.J,
        'K': Alphabet.K,
        'L': Alphabet.L,
        'Ł': Alphabet.Ł,
        'M': Alphabet.M,
        'N': Alphabet.N,
        'Ń': Alphabet.Ń,
        'O': Alphabet.O,
        'Ó': Alphabet.Ó,
        'P': Alphabet.P,
        'Q': Alphabet.Q,
        'R': Alphabet.R,
        'S': Alphabet.S,
        'Ś': Alphabet.Ś,
        'T': Alphabet.T,
        'U': Alphabet.U,
        'W': Alphabet.W,
        'V': Alphabet.V,
        'Y': Alphabet.Y,
        'Z': Alphabet.Z,
        'Ź': Alphabet.Ź,
        'Ż': Alphabet.Ż,
        'X': Alphabet.X,
        ',': Alphabet.COMMA,
        '*': Alphabet.STAR,
        'ST': Alphabet.STAR,
        '.': Alphabet.DOT,
        '!': Alphabet.EXCLAMATION_MARK,
        ' ': Alphabet.SPACE,
        'SP': Alphabet.SPACE,
        'NL': Alphabet.NEWLINE,
        '?': Alphabet.QUESTION_MARK,
        '#': Alphabet.HASH,
        'HS': Alphabet.HASH,
        'DEL': Alphabet.DEL,
        '@': Alphabet.AT,
        ':': Alphabet.COLON,
        ';': Alphabet.SEMICOLON,
        "'": Alphabet.QUOTE,
        '"': Alphabet.DOUBLE_QUOTE,
        '(': Alphabet.LEFT_PAREN,
        ')': Alphabet.RIGHT_PAREN,
        '[': Alphabet.LEFT_BRACKET,
        ']': Alphabet.RIGHT_BRACKET,
        '{': Alphabet.LEFT_BRACE,
        '}': Alphabet.RIGHT_BRACE,
        '<': Alphabet.LESS_THAN,
        '>': Alphabet.GREATER_THAN,
        '/': Alphabet.SLASH,
        '\\': Alphabet.BACKSLASH,
        '|': Alphabet.PIPE,
        '&': Alphabet.AMPERSAND,
        '%': Alphabet.PERCENT,
        '$': Alphabet.DOLLAR,
        '^': Alphabet.CARET,
        '~': Alphabet.TILDE,
        '_': Alphabet.UNDERSCORE,
        '+': Alphabet.PLUS,
        '-': Alphabet.MINUS,
        '=': Alphabet.EQUAL,

        'a': Alphabet.a,
        'ą': Alphabet.ą,
        'b': Alphabet.b,
        'c': Alphabet.c,
        'ć': Alphabet.ć,
        'd': Alphabet.d,
        'e': Alphabet.e,
        'ę': Alphabet.ę,
        'f': Alphabet.f,
        'g': Alphabet.g,
        'h': Alphabet.h,
        'i': Alphabet.i,
        'j': Alphabet.j,
        'k': Alphabet.k,
        'l': Alphabet.l,
        'ł': Alphabet.ł,
        'm': Alphabet.m,
        'n': Alphabet.n,
        'ń': Alphabet.ń,
        'o': Alphabet.o,
        'ó': Alphabet.ó,
        'q': Alphabet.q,
        'p': Alphabet.p,
        'r': Alphabet.r,
        's': Alphabet.s,
        'ś': Alphabet.ś,
        't': Alphabet.t,
        'u': Alphabet.u,
        'w': Alphabet.w,
        'v': Alphabet.v,
        'y': Alphabet.y,
        'z': Alphabet.z,
        'ź': Alphabet.ź,
        'ż': Alphabet.ż,
        'x': Alphabet.x,
        '0': Alphabet.ZERO,
        '1': Alphabet.ONE,
        '2': Alphabet.TWO,
        '3': Alphabet.THREE,
        '4': Alphabet.FOUR,
        '5': Alphabet.FIVE,
        '6': Alphabet.SIX,
        '7': Alphabet.SEVEN,
        '8': Alphabet.EIGHT,
        '9': Alphabet.NINE, 

        # special chars
        'QM': Alphabet.DOUBLE_QUOTE,
        'AP': Alphabet.QUOTE,
        'BS': Alphabet.BACKSLASH,
        'TB': Alphabet.SPACE,
        'BS': Alphabet.DEL,
        'DS': Alphabet.DOLLAR,

        # weird letters we wish we didn't have
        "è" : Alphabet.ę,
        "ë" : Alphabet.ę,
        "ö" : Alphabet.ó


    }
    e = char_map.get(char, None)
    if e is None:
        print(f"Unknown char {char}")
        print(f"Len char {len(char)}")

    return e

def char_to_enum_value(char):
    return char_to_enum(char).value

char_frequency = {
    'SP': 2756, 'DEL': 2036, 'e': 1279, 'a': 1262, 'i': 1109, 'o': 1064, 'n': 691, 'z': 683, 's': 636, 't': 627,
    'y': 519, 'm': 503, 'r': 495, 'l': 480, 'c': 460, 'k': 427, 'w': 418, 'd': 415, 'u': 344, 'p': 320, 'j': 317,
    'b': 266, 'h': 254, '.': 204, 'g': 191, ',': 117, 'f': 78, 'ł': 61, 'v': 50, 'ż': 36, 'x': 36, 'I': 32, 
    'ą': 30, 'M': 29, 'è': 27, 'NL': 26, 'W': 25, 'T': 24, 'ę': 23, 'ć': 23, 'ó': 18, 'N': 17, '?': 17, 'P': 15,
    'A': 15, '!': 15, 'ś': 14, 'H': 14, 'D': 14, 'B': 11, '(': 11, 'AP': 10, '1': 10, ')': 10, 'O': 9, 'R': 8,
    'q': 7, 'E': 7, 'C': 7, 'ź': 6, 'ń': 6, 'QM': 6, 'S': 5, 'J': 5, 'Z': 4, 'Y': 4, 'G': 4, ':': 4, '3': 4, 
    'F': 3, '8': 3, '6': 3, '5': 3, '0': 3, 'U': 2, '9': 2, '{': 1, 'V': 1, 'K': 1, 'DS': 1, '7': 1, '2': 1, 
    '-': 1, '#': 1
}

# nobody wrote a pipe
weird_char_coded = Alphabet.PIPE.value 
def char_to_enum_value_without_uncommon(char, cutoff=10):
    if char_frequency.get(char, 0) <= cutoff:
        return weird_char_coded 
    return char_to_enum(char).value