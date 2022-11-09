import pressio
import numpy as np
from numcodecs.abc import Codec
from numcodecs.compat import ndarray_copy
if pressio.LIBPRESSIO_HAS_MPI4PY:
    from mpi4py import MPI
import ctypes

try:
    lib = ctypes.cdll.LoadLibrary("liblibpressio_meta.so")
    lib.libpressio_register_all()
except OSError:
    pass

def supported_io():
    """returns the list of valid io modules"""
    supported = pressio.supported_io_modules()
    return [s.decode() for s in supported.split(b' ') if s]


def supported_compressors():
    """returns the list of valid compressor modules"""
    supported = pressio.supported_compressors()
    return [s.decode() for s in supported.split(b' ') if s]


def supported_metrics():
    """returns the list of valid compressor modules"""
    supported = pressio.supported_metrics()
    return [s.decode() for s in supported.split(b' ') if s]


class PressioException(Exception):
    """represents and error for a libpressio object"""
    def __init__(self, msg, error_code):
        self.msg = msg
        self.error_code = error_code
        super().__init__(msg)

    @classmethod
    def from_compressor(cls, compressor):
        """represents and error for a libpressio compressor object"""
        msg = pressio.compressor_error_msg(compressor).decode()
        error_code = pressio.compressor_error_code(compressor)
        return cls(msg, error_code)

    @classmethod
    def from_library(cls, library):
        """represents and error for a libpressio library object"""
        msg = pressio.error_msg(library).decode()
        error_code = pressio.error_code(library)
        return cls(msg, error_code)

    @classmethod
    def from_io(cls, io):
        """represents and error for a libpressio io object"""
        msg = pressio.io_error_msg(io).decode()
        error_code = pressio.io_error_code(io)
        return cls(msg, error_code)

def _python_to_pressio(options, template=None):
    config = pressio.options_new() 

    def _compute_path(path, key):
        if len(path) > 1:
            ret = b'/'.join(p.encode() for p in path) + b":" + key.encode()
        else:
            ret = key.encode()
        return ret

    def to_libpressio(x):
        op = None
        if isinstance(value, np.ndarray):
            value_lp = pressio.io_data_from_numpy(value)
            op = pressio.option_new_data(value_lp)
            pressio.data_free(value_lp)
        elif isinstance(value, list):
            if value:
                # list is non-empty
                if isinstance(value[0], str):
                    op = pressio.option_new_strings(pressio.vector_string([i.encode() for i in value]))
                elif isinstance(value[0], int) or isinstance(value[0], float):
                    arr = np.array(value)
                    lp = pressio.io_data_from_numpy(arr)
                    op = pressio.option_new_data(lp)
                    pressio.data_free(lp)
                else:
                    raise TypeError("unexpected list type: " + value)
            else:
                # list is empty
                op = pressio.option_new_strings(pressio.vector_string())
        elif isinstance(value, float):
            op = pressio.option_new_double(value)
        elif isinstance(value, str):
            op = pressio.option_new_string(value.encode())
        elif isinstance(value, bytes):
            data = pressio.io_data_from_bytes(value)
            op = pressio.option_new_data(data)
            pressio.data_free(data)
        elif isinstance(value, bool):
            op = pressio.option_new_bool(value)
        elif isinstance(value, int):
            op = pressio.option_new_integer64(value)
        elif pressio.LIBPRESSIO_HAS_MPI4PY:
            if isinstance(value, MPI.Comm):
                op = pressio.option_new_comm(value)
        else:
            raise TypeError("Unsupported type " + str(type(value)))
        return op

    entries = [([''], options)]
    while entries:
        path, entry = entries.pop()
        for key, value in entry.items():
            op = None
            try:
                if isinstance(value, dict):
                    entries.append((path + [key], value))
                    continue
                elif value is None:
                    pass
                else:
                    op = to_libpressio(value)

                name = _compute_path(path, key)
                if template is not None:
                    status = pressio.options_exists(template, name)

                    if status == pressio.options_key_set or status == pressio.options_key_exists:
                        try:
                            op_template = pressio.options_get(template, name)
                            if op is not None:
                                # try to preform a conversion if we can
                                status = pressio.option_cast_set(op_template, op,
                                                                 pressio.conversion_special)
                                if status == pressio.options_key_set:
                                    pressio.options_set(config, name, op_template)
                                elif status == pressio.options_key_exists:
                                    raise TypeError("invalid type for " + key)
                                elif status == pressio.options_key_does_not_exist:
                                    raise TypeError("does not exist: " + key)
                            else:
                                # we have a type, so pass along that information
                                template_type = pressio.option_get_type(op_template)
                                pressio.options_set_type(config, name, template_type)
                        finally:
                            pressio.option_free(op_template)
                    elif status == pressio.options_key_does_not_exist:
                        raise TypeError("does not exist: " + key)
                elif op is not None:
                    pressio.options_set(config, name, op)
            finally:
                pressio.option_free(op)

    return config


def _no_conversion(x):
    raise TypeError("no conversion")


def _from_data_option(x):
    lp = pressio.option_get_data(x)
    if pressio.data_dtype(lp) == pressio.byte_dtype:
        ret = pressio.io_data_to_bytes(lp)
    else:
        ret = pressio.io_data_to_numpy(lp)
    pressio.data_free(lp)
    return ret


def _from_charptr_array(x):
    return list(i.decode() for i in pressio.option_get_strings(x))


_config_to_option_converters = {
    pressio.option_dtype_type: pressio.option_get_dtype,
    pressio.option_threadsafety_type: pressio.option_get_threadsafety,
    pressio.option_bool_type: pressio.option_get_bool,
    pressio.option_int8_type: pressio.option_get_integer8,
    pressio.option_int16_type: pressio.option_get_integer16,
    pressio.option_int32_type: pressio.option_get_integer,
    pressio.option_int64_type: pressio.option_get_integer64,
    pressio.option_uint8_type: pressio.option_get_uinteger8,
    pressio.option_uint16_type: pressio.option_get_uinteger16,
    pressio.option_uint32_type: pressio.option_get_uinteger,
    pressio.option_uint64_type: pressio.option_get_uinteger64,
    pressio.option_float_type: pressio.option_get_float,
    pressio.option_double_type: pressio.option_get_double,
    pressio.option_charptr_type: lambda x: pressio.option_get_string(x).decode(),
    pressio.option_charptr_array_type: _from_charptr_array,
    pressio.option_data_type: _from_data_option,
    pressio.option_userptr_type: _no_conversion,
    pressio.option_unset_type: _no_conversion,
}


def _pressio_to_python(config):
    def to_python(value):
        if not pressio.option_has_value(value):
            return None
        return _config_to_option_converters[pressio.option_get_type(value)](value)

    options = {}

    def set_option(key, value):
        t = options
        parts = key.split(b':')
        if not key.startswith(b'/'):
            # global option
            t[key.decode()] = value
        else:
            # hierarchical option
            paths = [i.decode() for i in parts[0].split(b'/')]
            option_name = b":".join(parts[1:])
            for path in paths[1:]:
                if path not in t:
                    t[path] = {}
                t = t[path]
            t[option_name.decode()] = value

    iter = pressio.options_get_iter(config)
    while pressio.options_iter_has_value(iter):
        key = pressio.options_iter_get_key(iter)
        value = pressio.options_iter_get_value(iter)
        try:
            set_option(key, to_python(value))
        except TypeError:
            pass
        pressio.option_free(value)

        pressio.options_iter_next(iter)
    pressio.options_iter_free(iter)
    return options


class PressioCompressor(Codec):
    """wrapper for a libpressio compressor object"""

    def __init__(self, compressor_id="noop", early_config={}, compressor_config={}, name=""):
        """wrapper for a libpressio compressor object

        params:
            compressor_id: str - the compressor_id for the underlying compressor
            early_config: dict - converted to pressio_options to configure the structure of the compressor
            compressor_config: dict - converted to pressio_options to configure the compressor
            name: str - name to use for the compressor when used in a hierarchical mode
        """
        try:
            library = pressio.instance()
            self._name = name
            self._compressor_id = compressor_id
            self._compressor = pressio.get_compressor(library, compressor_id.encode())
            if self._compressor is None:
                raise PressioException.from_library(library)

            if name:
                pressio.compressor_set_name(self._compressor, name.encode())

            early_config_lp = _python_to_pressio(early_config)
            ec = pressio.compressor_set_options(self._compressor, early_config_lp)
            pressio.options_free(early_config_lp)
            if ec != 0:
                raise PressioException.from_compressor(self._compressor)

            config_lp_template = pressio.compressor_get_options(self._compressor)
            config_lp = _python_to_pressio(compressor_config, config_lp_template)
            ec = pressio.compressor_set_options(self._compressor, config_lp)
            pressio.options_free(config_lp)
            pressio.options_free(config_lp_template)
            if ec != 0:
                raise PressioException.from_compressor(self._compressor)

        finally:
            pressio.release(library)

    def __del__(self):
        # works around a bug in swig 4.0 where during shutdown pressio is set to None
        # indicating that the shared library has been unloaded.  Just let the OS
        # clean up in this case.
        if pressio:
            pressio.compressor_release(self._compressor)

    def encode(self, uncompressed):
        """perform compression

        params:
            uncompressed: np.ndarray - the data to be compressed
        """
        uncompressed_lp = None
        compressed_lp = None
        try:
            uncompressed_lp = pressio.io_data_from_numpy(uncompressed)
            compressed_lp = pressio.data_new_empty(pressio.byte_dtype, pressio.vector_uint64_t())

            rc = pressio.compressor_compress(self._compressor, uncompressed_lp, compressed_lp)
            if rc:
                raise PressioException.from_compressor(self._compressor)

            comp = pressio.io_data_to_numpy(compressed_lp)
        finally:
            pressio.data_free(uncompressed_lp)
            pressio.data_free(compressed_lp)
        return comp

    def decode(self, compressed, decompressed=None):
        """perform decompression

        params:
            compressed: bytes - the data to be decompressed
            decompressed: numpy.ndarray - memory to be used to decompress data into
        """
        compressed_lp = None
        decompressed_lp = None
        try:
            compressed_lp = pressio.io_data_from_numpy(compressed)
            decompressed_lp = pressio.io_data_from_numpy(decompressed)

            rc = pressio.compressor_decompress(self._compressor, compressed_lp, decompressed_lp)
            if rc:
                raise PressioException.from_compressor(self._compressor)

            dec = pressio.io_data_to_numpy(decompressed_lp)

            if decompressed is not None:
                return dec
            else:
                return ndarray_copy(dec, decompressed)
        finally:
            pressio.data_free(compressed_lp)
            pressio.data_free(decompressed_lp)

    def get_compile_config(self):
        """get compile time configuration"""
        lp_options = pressio.compressor_get_configuration(self._compressor)
        options = _pressio_to_python(lp_options)
        pressio.options_free(lp_options)
        return options

    def get_metrics(self):
        """get runtime time metrics"""
        lp_options = pressio.compressor_get_metrics_results(self._compressor)
        options = _pressio_to_python(lp_options)
        pressio.options_free(lp_options)
        return options

    def _get_config(self):
        lp_options = pressio.compressor_get_options(self._compressor)
        options = _pressio_to_python(lp_options)
        pressio.options_free(lp_options)
        return options

    def get_config(self):
        """get runtime time options"""
        options = self._get_config()
        return {
            "id": self.codec_id,
            "compressor_id": self._compressor_id,
            "early_config": options,
            "compressor_config": options,
            "name": pressio.compressor_get_name(self._compressor).decode()
        }

    def set_config(self, config):
        """set runtime time options"""
        try:
            options = _python_to_pressio(config)
            if pressio.compressor_set_options(self._compressor, options):
                raise PressioCompressor.from_compressor(self._compressor)
        finally:
            pressio.options_free(options)

    @staticmethod
    def _recursive_keys(config):
        entries = [config]
        while entries:
            entry = entries.pop()
            for key, value in entry.items():
                if isinstance(value, dict):
                    entries.append(value)
                yield key

    @property
    def codec_id(self):
        """returns a unique key based on this structure of compressor"""
        return "pressio:" + str(hash(frozenset(self._recursive_keys(self._get_config()))))

    @classmethod
    def from_config(cls, config):
        """returns a compressor from a configuration

        params:
            config: dict - the configuration to build the compressor from
                compressor_id: str - the id of the compressors to load
                early_config: Optional[dict] - the structural configuration of the compressor or nothing
                compressor_config: Optional[dict] - the configuration of the compressor or nothing
                name: Optional[str] - the name of the compressor if one is to be used
        """
        return cls(config['compressor_id'],
                   config.get('early_config', {}),
                   config.get('compressor_config', {}),
                   config.get('name', None)
                   )


class PressioIO:

    def __init__(self, io, early_config, io_config, name):
        try:
            config_lp = None
            library = pressio.instance()
            self._io = pressio.get_io(library, io.encode())
            self._io_id = io
            if not self._io:
                raise PressioException.from_library(library)

            if name is not None:
                pressio.compressor_set_name(self._io, name.encode())

            early_config_lp = _python_to_pressio(early_config)
            pressio.io_set_options(self._io, early_config_lp)
            pressio.options_free(early_config_lp)

            config_lp_template = pressio.io_get_options(self._io)
            config_lp = _python_to_pressio(io_config, config_lp_template)

            pressio.io_set_options(self._io, config_lp)
        finally:
            pressio.release(library)
            pressio.options_free(config_lp)
            pressio.options_free(config_lp_template)

    def __del__(self):
        # works around a bug in swig 4.0 where during shutdown pressio is set to None
        # indicating that the shared library has been unloaded.  Just let the OS
        # clean up in this case.
        if pressio:
            pressio.io_free(self._io)

    def read(self, template=None):
        """reads a data buffer from a file

        params:
            template: Optional[np.ndarray] - a input template if one is provided
        """
        if template is not None:
            template = pressio.io_data_from_numpy(template)
        ret_lp = pressio.io_read(self._io, template)
        if not ret_lp:
            raise PressioException.from_io(self._io)
        ret = pressio.io_data_to_numpy(ret_lp)
        pressio.data_free(ret_lp)
        if template is not None:
            pressio.data_free(template)
        return ret

    def write(self, output):
        """reads a data buffer to a file

        params:
            output: np.ndarray - the file to be written
        """
        try:
            out = pressio.io_data_from_numpy(output)
            ret = pressio.io_write(self._io, out)
            if ret:
                raise PressioException.from_io(self._io)
            return ret
        finally:
            pressio.data_free(out)

    def _get_config(self):
        lp_options = pressio.io_get_options(self._io)
        options = _pressio_to_python(lp_options)
        pressio.options_free(lp_options)
        return options

    def get_config(self):
        """get runtime configuration"""
        options = self._get_config()
        return {
            "io_id": self._io_id,
            "early_config": options,
            "io_config": options,
            "name": pressio.io_get_name(self._io).decode()
        }

    def get_compile_config(self):
        """get compile time configuration"""
        lp_options = pressio.io_get_configuration(self._io)
        options = _pressio_to_python(lp_options)
        pressio.options_free(lp_options)
        return options

    def set_config(self, config):
        """set runtime time options"""
        try:
            options = _python_to_pressio(config)
            if pressio.io_set_options(self._io, options):
                raise PressioException.from_io(self._io)
        finally:
            pressio.options_free(options)

    @classmethod
    def from_config(cls, config):
        """returns a compressor from a configuration

        params:
            config: dict - the configuration to build the io object from
                io_id: str - the id of the io object to load
                early_config: Optional[dict] - the structural configuration of the io object or nothing
                io_config: Optional[dict] - the configuration of the io object or nothing
                name: Optional[str] - the name of the io object if one is to be used
        """
        return cls(config['io_id'],
                   config.get('early_config', {}),
                   config.get('io_config', {}),
                   config.get('name', None)
                   )


class PressioMetrics:
    @classmethod
    def from_config(cls, config):
        return cls(config.get('metric_ids', []),
                   config.get('early_config', {}),
                   config.get('metrics_config', {}),
                   config.get('name', None)
                   )

    def __init__(self, ids, early_config, metrics_config, name):
        try:
            config_lp = None
            library = pressio.instance()
            metrics_ids = pressio.vector_string([i.encode() for i in ids])
            self._metric = pressio.new_metrics(library, metrics_ids)
            self._metric_id = "composite"
            if not self._metric:
                raise PressioException.from_library(library)

            if name is not None:
                pressio.metrics_set_name(self._metric, name.encode())

            early_config_lp = _python_to_pressio(early_config)
            pressio.metrics_set_options(self._metric, early_config_lp)
            pressio.options_free(early_config_lp)

            config_lp_template = pressio.metrics_get_options(self._metric)
            config_lp = _python_to_pressio(metrics_config, config_lp_template)

            pressio.metrics_set_options(self._metric, config_lp)
        finally:
            pressio.release(library)
            pressio.options_free(config_lp)
            pressio.options_free(config_lp_template)

    def __del__(self):
        # works around a bug in swig 4.0 where during shutdown pressio is set to None
        # indicating that the shared library has been unloaded.  Just let the OS
        # clean up in this case.
        if pressio:
            pressio.metrics_free(self._metric)

    def evaluate(self, input=None, compressed=None, output=None):
        results_lp = None
        try:
            input_lp = input if input is None else pressio.io_data_from_numpy(input)
            compressed_lp = compressed if compressed is None else pressio.io_data_from_bytes(compressed)
            output_lp = output if output is None else pressio.io_data_from_numpy(output)
            results_lp = pressio.metrics_evaluate(self._metric, input_lp, compressed_lp, output_lp)
            return _pressio_to_python(results_lp)
        finally:
            if input_lp is not None:
                pressio.data_free(input_lp)
            if output_lp is not None:
                pressio.data_free(output_lp)
            if compressed_lp is not None:
                pressio.data_free(compressed_lp)
            if results_lp is not None:
                pressio.options_free(results_lp)


    def set_config(self, config):
        """set runtime time options"""
        try:
            options = _python_to_pressio(config)
            pressio.metrics_set_options(options)
        finally:
            pressio.options_free(options)

    @staticmethod
    def _recursive_keys(config):
        entries = [config]
        while entries:
            entry = entries.pop()
            for key, value in entry.items():
                if isinstance(value, dict):
                    entries.append(value)
                yield key

    @property
    def codec_id(self):
        """returns a unique key based on this structure of compressor"""
        return "pressio:" + str(hash(frozenset(self._recursive_keys(self._get_config()))))


    def _get_config(self):
        lp_options = pressio.metrics_get_options(self._metric)
        options = _pressio_to_python(lp_options)
        pressio.options_free(lp_options)
        return options


    def get_config(self):
        """get runtime time options"""
        options = self._get_config()
        return {
            "id": self.codec_id,
            "metric_id": self._metric_id,
            "early_config": options,
            "metrics_config": options,
            "name": pressio.metrics_get_name(self._metric).decode()
        }
