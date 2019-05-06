from ID3 import Dataset, Node
import pandas as pd
import numpy as np
import sys, os

if __name__ == '__main__':
    import sys, os

    if not sys.argv[1]:
        print("Usage: python3 id3.py [filename.txt]")

    elif not os.path.isfile(sys.argv[0]):
        print("File does not exist.")
        print("Usage: python3 id3.py [filename.txt]")

    else:
        filename = sys.argv[1]
        with open(filename) as fd:
            f = fd.readlines()
        d = Dataset(f)
        print("")
        d.build_tree().print()
        print("\nTraining set accuracy: {0:.2f} %".format(d.accuracy()*100))