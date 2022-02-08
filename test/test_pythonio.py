import sys
import tempfile
#kind of ugly hack to load the path to the library without installing it
#the path gets passed from the CMakeLists.txt file
pressio_path = sys.argv[1]
sys.path.insert(0, pressio_path)

import numpy as np
np.random.seed(0)

import pressio

library = pressio.instance()
compressor = pressio.get_compressor(library, b"sz")
sz_options = pressio.compressor_get_options(compressor)
metrics_ids = pressio.vector_string([b'time',b'size'])
metrics = pressio.new_metrics(library, metrics_ids)

pressio.options_set_string(sz_options, b"sz:error_bound_mode_str", b'abs');
pressio.options_set_double(sz_options, b"sz:abs_err_bound", 0.5);

pressio.compressor_check_options(compressor, sz_options)
pressio.compressor_set_options(compressor, sz_options)
pressio.compressor_set_metrics(compressor, metrics)

data = np.random.rand(300, 300, 300)
with tempfile.NamedTemporaryFile() as tmp:
    data.tofile(tmp)

    posix_io = pressio.get_io(library, b'posix')

    #configure read
    posix_options = pressio.io_get_options(posix_io)
    pressio.options_set_string(posix_options, b'io:path', bytes(tmp.name, 'ascii'))
    pressio.io_set_options(posix_io, posix_options)

    #configure template buffer
    dims = pressio.vector_uint64_t([300,300,300])
    input_template = pressio.data_new_empty(pressio.double_dtype, dims)

    #preform read
    input_data = pressio.io_read(posix_io, input_template)
    

    compressed_data = pressio.data_new_empty(pressio.byte_dtype, pressio.vector_uint64_t())
    decompressed_data = pressio.data_new_empty(pressio.double_dtype, dims)

    pressio.compressor_compress(compressor, input_data, compressed_data)

    pressio.compressor_decompress(compressor, compressed_data, decompressed_data)

    metric_results = pressio.compressor_get_metrics_results(compressor)
    compression_ratio = pressio.new_double()
    pressio.options_get_double(metric_results, b"size:compression_ratio", compression_ratio)
    print("compression ratio", pressio.double_value(compression_ratio))

    pressio.io_free(posix_io)
    pressio.options_free(posix_options)
    pressio.options_free(metric_results)
    pressio.options_free(sz_options)
    pressio.metrics_free(metrics)
    pressio.data_free(input_template)
    pressio.data_free(input_data)
    pressio.data_free(decompressed_data)
    pressio.data_free(compressed_data)
    pressio.delete_double(compression_ratio)
    pressio.compressor_release(compressor)
    pressio.release(library)

