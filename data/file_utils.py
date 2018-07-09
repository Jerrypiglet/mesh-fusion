import os
import termcolor

# create directory
def mkdir(path):
    if not os.path.exists(path): os.mkdir(path)

# convert to colored strings
def toRed(content): return termcolor.colored(content,"red",attrs=["reverse", "blink"])
def toGreen(content): return termcolor.colored(content,"green",attrs=["reverse", "blink"])
def toBlue(content): return termcolor.colored(content,"blue",attrs=["reverse", "blink"])
def toCyan(content): return termcolor.colored(content,"cyan",attrs=["reverse", "blink"])
def toYellow(content): return termcolor.colored(content,"yellow",attrs=["reverse", "blink"])
def toMagenta(content): return termcolor.colored(content,"magenta",attrs=["bold", "blink"])
