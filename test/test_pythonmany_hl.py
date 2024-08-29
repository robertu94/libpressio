import libpressio
import numpy as np
from pprint import pprint
np.random.seed(0)

input1 = np.random.rand(300,300,300)
input2 = np.random.rand(300,300,300)
inputs = [input1, input2]
output1 = input1.copy()
output2 = input2.copy()
outputs = [output1, output2]

compressor = libpressio.PressioCompressor("many_independent_threaded",
                                          early_config={
                                              "many_independent_threaded:compressor": b"sz3",
                                              "many_independent_threaded:metric": b"size"
                                              },
                                          compressor_config={
                                              'many_independent_threaded:collect_metrics_on_compression': 0,
                                              'many_independent_threaded:collect_metrics_on_decompression': 1,
                                              'many_independent_threaded:preserve_metrics': 1,
                                              }
                                          )


compressed = compressor.encode_many(inputs)
outputs = compressor.decode_many(compressed, outputs)

pprint(compressor.metrics_results())
