import libpressio
import pressio
import numpy as np
from pprint import pprint
np.random.seed(0)

input1 = np.random.rand(300,300,300)
input2 = np.random.rand(300,300,300)
output1 = input1.copy()
output2 = input2.copy()

library = pressio.instance()

compressor = pressio.get_compressor(library, b"many_independent_threaded")
options = pressio.options_new()
pressio.options_set_string(options, b"many_independent_threaded:compressor", b"sz3")
pressio.options_set_string(options, b"many_independent_threaded:metric", b"size")
pressio.options_set_integer(options, b'many_independent_threaded:collect_metrics_on_compression', 0)
pressio.options_set_integer(options, b'many_independent_threaded:collect_metrics_on_decompression', 1)
pressio.options_set_integer(options, b'many_independent_threaded:preserve_metrics', 1)
pressio.options_set_string(options, b"sz3:metric", b"historian")
pressio.options_set_string(options, b"historian:metrics", b"external")
pressio.options_set_strings(options, b"historian:events", pressio.vector_string([b"decompress_many", b"clone"]))
pressio.options_set_double(options, b"pressio:abs", 1e-3)
pressio.compressor_set_options(compressor, options)
pressio.options_free(options)

options = pressio.compressor_get_options(compressor)
pprint(libpressio.pressio_options_to_python(options))
pressio.options_free(options)

p_input1 = libpressio.python_to_pressio_data(input1)
p_input2 = libpressio.python_to_pressio_data(input2)
p_inputs = pressio.vector_data()
p_inputs.append(p_input1)
p_inputs.append(p_input2)

p_compressed1 = pressio.data_new_empty(pressio.byte_dtype, pressio.vector_uint64_t())
p_compressed2 = pressio.data_new_empty(pressio.byte_dtype, pressio.vector_uint64_t())
p_compresseds = pressio.vector_data()
p_compresseds.append(p_compressed1)
p_compresseds.append(p_compressed2)



rc = pressio.compressor_compress_many(compressor, p_inputs, p_compresseds)
if rc != 0:
    raise libpressio.PressioException.from_compressor(compressor)
pressio.data_free(p_input1)
pressio.data_free(p_input2)

p_output1 = libpressio.python_to_pressio_data(output1)
p_output2 = libpressio.python_to_pressio_data(output2)
p_outputs = pressio.vector_data()
p_outputs.append(p_output1)
p_outputs.append(p_output2)

pressio.compressor_decompress_many(compressor, p_compresseds, p_outputs)
if rc != 0:
    raise libpressio.PressioException.from_compressor(compressor)

pressio.data_free(p_compressed1)
pressio.data_free(p_compressed2)
pressio.data_free(p_output1)
pressio.data_free(p_output2)

p_metrics = pressio.compressor_get_metrics_results(compressor)
metrics = libpressio.pressio_options_to_python(p_metrics)
pprint(metrics)
pressio.options_free(p_metrics)
