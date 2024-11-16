import libpressio
import numpy as np
from pprint import pprint
np.random.seed(0)

input1 = np.random.rand(300,300,300)
input2 = np.random.rand(300,300,300)
inputs = [np.random.rand(300,300,300), np.random.rand(300,300,300)]
outputs = [inputs[0].copy(), inputs[1].copy()]

compressor = libpressio.PressioCompressor("many_independent_threaded",
                                          early_config={
                                              "many_independent_threaded:compressor": "sz3",
                                              "many_independent_threaded:metric": "size",
                                              "sz3:metric": "historian",
                                              "historian:events": ["decompress_many", "clone"],
                                              "historian:metrics": "size",
                                              },
                                          compressor_config={
                                              'many_independent_threaded:collect_metrics_on_compression': 0,
                                              'many_independent_threaded:collect_metrics_on_decompression': 1,
                                              'many_independent_threaded:preserve_metrics': 1,
                                              }
                                          )

pprint(compressor.get_options())

compressed = compressor.encode_many(inputs)
outputs = compressor.decode_many(compressed, outputs)

pprint(compressor.get_metrics())
