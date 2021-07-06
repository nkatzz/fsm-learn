import string
import clingo


def symbols(symbol_index):
    alphabet = list(string.ascii_lowercase)
    return alphabet[symbol_index - 1]



print(symbols(20))
