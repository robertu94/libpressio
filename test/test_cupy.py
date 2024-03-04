#!/usr/bin/env python

import sys
pressio_path = sys.argv[1]
sys.path.insert(0, pressio_path)
from pprint import pprint
import sys
import libpressio as lp
try:
    import cupy as cp
except ImportError:
    print("cupy does not exist, skipping")
    sys.exit(0)

x = cp.arange(0,100)
y = cp.arange(0,100)

# test with provided output location
input = cp.outer(x,y)
output = input.copy()
compressed = input.copy()
comp = lp.PressioCompressor.from_config({
        "compressor_id":"zfp",
        "early_config": {
            "zfp:metric": "composite",
            "composite:plugins": ["size", "time"]
        },
        "compressor_config": {
            "zfp:rate": 16,
            "zfp:execution_name": "cuda"
        }
    });
compressed = comp.encode(input, out=compressed)
output = comp.decode(compressed, output)
pprint(comp.get_metrics())

# test with allocated memory location
#input = cp.outer(x,y)
#output = input.copy()
#compressed = input.copy()
#comp = lp.PressioCompressor.from_config({
#        "compressor_id":"zfp",
#        "early_config": {
#            "zfp:metric": "composite",
#            "composite:plugins": ["size", "time"]
#        },
#        "compressor_config": {
#            "zfp:rate": 16,
#            "zfp:execution_name": "cuda"
#        }
#    });
#compressed = comp.encode(input)
#output = comp.decode(compressed, output)
#pprint(comp.get_metrics())
