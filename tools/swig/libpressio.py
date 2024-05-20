"""
Highlevel Python Bindings for LibPressio

Documentation: https://robertu94.github.io/libpressio
Tutorial: https://github.com/robertu94/libpressio_tutorial
"""
import pressio
import numpy as np
import abc
import os
from numcodecs.abc import Codec
from numcodecs.compat import ndarray_copy
if pressio.LIBPRESSIO_HAS_MPI4PY:
    from mpi4py import MPI
import ctypes

try:
    _lib = ctypes.cdll.LoadLibrary(os.environ.get("LIBPRESSIO_PLUGINS", "liblibpressio_meta.so"))
    _lib.libpressio_register_all()
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

    @classmethod
    def from_metric(cls, metrics):
        """represents and error for a libpressio io object"""
        msg = pressio.metrics_error_msg(metrics).decode()
        error_code = pressio.metrics_error_code(metrics)
        return cls(msg, error_code)

def python_to_pressio_data(x):
    if hasattr(x, "__cuda_array_interface__"):
        info = x.__cuda_array_interface__
        if info.get("strides",None) is not None:
            raise NotImplementedError("stridded cuda arrays are not supported")
        if info.get("mask", None) is not None:
            raise NotImplementedError("masked cuda arrays are not supported")
        if info['version'] != 3:
            raise NotImplementedError("only version 3 is supported")
        return pressio.io_data_from_cuda_array(
                pressio.vector_uint64_t(info['shape']),
                info['typestr'].encode(),
                info['data'][0]
                )
    elif isinstance(x, np.ndarray):
        return pressio.io_data_from_numpy(x)
    else:
        raise NotImplementedError()

def pressio_data_to_python(x, out=None):
    domain = pressio.data_domain_id(x)
    if domain == b"cudamalloc":
        info = pressio.io_data_to_cuda_array(x)
        class PressioDataCuda:
            def __init__(self, x, info):
                self.ptr = x
                self.__cuda_array_interface__ = {
                    "shape": tuple(info.shape),
                    "typestr": info.typestr.decode(),
                    "data": (info.ptr, info.read_only),
                    "version": info.version
                }
            def __del__(self):
                pressio.data_free(self.ptr)
        return PressioDataCuda(x, info), False
    else:
        ret = pressio.io_data_to_numpy(x)
        if out is not None:
            ret = ndarray_copy(ret, out)
    return ret, True

def python_to_pressio_options(options, template=None):
    config = pressio.options_new()

    def _compute_path(path, key):
        if len(path) > 1:
            ret = b'/'.join(p.encode() for p in path) + b":" + key.encode()
        else:
            ret = key.encode()
        return ret

    def to_libpressio_option(x):
        op = None
        if isinstance(x, np.ndarray):
            value_lp = python_to_pressio_data(x)
            op = pressio.option_new_data(value_lp)
            pressio.data_free(value_lp)
        elif isinstance(x, list):
            if x:
                # list is non-empty
                if isinstance(x[0], str):
                    op = pressio.option_new_strings(pressio.vector_string([i.encode() for i in x]))
                elif isinstance(x[0], int) or isinstance(x[0], float):
                    arr = np.array(x)
                    lp = python_to_pressio_data(arr)
                    op = pressio.option_new_data(lp)
                    pressio.data_free(lp)
                else:
                    raise TypeError("unexpected list type: " + str(x))
            else:
                # list is empty
                op = pressio.option_new_strings(pressio.vector_string())
        elif isinstance(x, float):
            op = pressio.option_new_double(x)
        elif isinstance(x, str):
            op = pressio.option_new_string(x.encode())
        elif isinstance(x, bytes):
            data = pressio.io_data_from_bytes(x)
            op = pressio.option_new_data(data)
            pressio.data_free(data)
        elif isinstance(x, bool):
            op = pressio.option_new_bool(x)
        elif isinstance(x, int):
            op = pressio.option_new_integer64(x)
        elif np.issubdtype(x, np.int8):
            op = pressio.option_new_integer8(int(x))
        elif np.issubdtype(x, np.int16):
            op = pressio.option_new_integer16(int(x))
        elif np.issubdtype(x, np.int32):
            op = pressio.option_new_integer(int(x))
        elif np.issubdtype(x, np.int64):
            op = pressio.option_new_integer64(int(x))
        elif np.issubdtype(x, np.uint8):
            op = pressio.option_new_uinteger8(int(x))
        elif np.issubdtype(x, np.uint16):
            op = pressio.option_new_uinteger16(int(x))
        elif np.issubdtype(x, np.uint32):
            op = pressio.option_new_uinteger(int(x))
        elif np.issubdtype(x, np.uint64):
            op = pressio.option_new_uinteger64(int(x))
        elif pressio.LIBPRESSIO_HAS_MPI4PY:
            if isinstance(x, MPI.Comm): # type: ignore
                op = pressio.option_new_comm(x)
        else:
            raise TypeError("Unsupported type " + str(type(x)))
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
                    op = to_libpressio_option(value)

                name = _compute_path(path, key)
                if template is not None:
                    status = pressio.options_exists(template, name)

                    if status == pressio.options_key_set or status == pressio.options_key_exists:
                        op_template = None
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


def _no_conversion(_x):
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


def pressio_options_to_python(config):
    def to_python_option(value):
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
            set_option(key, to_python_option(value))
        except TypeError:
            pass
        pressio.option_free(value)

        pressio.options_iter_next(iter)
    pressio.options_iter_free(iter)
    return options


class PressioObject(metaclass=abc.ABCMeta):
    def __init__(self, object, object_id, name):
        self._object = object
        self._object_id = object_id
        self._name = name

    @staticmethod
    @abc.abstractmethod
    def _typename():
        raise NotImplementedError()

    @abc.abstractmethod
    def get_name(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def from_object(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_configuration_impl(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_options_impl(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_documentation_impl(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def _set_options_impl(self, _options):
        raise NotImplementedError()

    def get_compile_config(self):
        """get compile time configuration"""
        lp_options = self._get_configuration_impl()
        options = pressio_options_to_python(lp_options)
        pressio.options_free(lp_options)
        return options

    def get_configuration(self):
        """get compile time configuration"""
        return self.get_compile_config()

    def get_documentation(self):
        """get documentation"""
        lp_options = self._get_documentation_impl()
        options = pressio_options_to_python(lp_options)
        pressio.options_free(lp_options)
        return options

    def get_options(self):
        """get runtime time configuration"""
        lp_options = self._get_options_impl()
        options = pressio_options_to_python(lp_options)
        pressio.options_free(lp_options)
        return options

    def get_config(self):
        """get runtime time options in numcodecs compatiable format"""
        options = self.get_options()
        return {
            "id": self.codec_id,
            "{}_id".format(self._typename()): self._object_id,
            "early_config": options,
            "{}_config".format(self._typename()): options,
            "name": self.get_name()
        }

    def set_options(self, config):
        """set runtime options; libpressio-style interface"""
        return self.set_config(config)

    def set_config(self, config):
        """set runtime time options; numcodecs compatible interface"""
        options = None
        try:
            options = python_to_pressio_options(config)
            if self._set_options_impl(options):
                raise self.from_object()(self._object)
        finally:
            pressio.options_free(options)

    def json(self):
        instance = None
        options = None
        json = None
        try:
            instance = pressio.instance()
            options = self._get_options_impl()
            json = pressio.options_to_json(instance, options)
            return json.decode()
        finally:
            pressio.options_free(options)
            pressio.release(instance)

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
        """returns a unique key based on this structure of object"""
        return "pressio:" + str(hash(frozenset(self._recursive_keys(self.get_options()))))

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
        return cls(config['{}_id'.format(cls._typename())],
                   config.get('early_config', {}),
                   config.get('{}_config'.format(cls._typename()), {}),
                   config.get('name', None)
                   )


class PressioCompressor(PressioObject, Codec):
    """wrapper for a libpressio compressor object"""

    @staticmethod
    def _typename():
        return "compressor"

    def get_name(self):
        return pressio.compressor_get_name(self._object).decode()

    def from_object(self):
        return PressioException.from_compressor

    def _get_configuration_impl(self):
        return pressio.compressor_get_configuration(self._object)

    def _get_options_impl(self):
        return pressio.compressor_get_options(self._object)

    def _get_documentation_impl(self):
        return pressio.compressor_get_documentation(self._object)

    def _set_options_impl(self, options):
        return pressio.compressor_set_options(self._object, options)


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
            obj_id = compressor_id
            obj = pressio.get_compressor(library, compressor_id.encode())
            if obj is None:
                raise PressioException.from_library(library)

            if name:
                pressio.compressor_set_name(obj, name.encode())

            early_config_lp = python_to_pressio_options(early_config)
            ec = pressio.compressor_set_options(obj, early_config_lp)
            pressio.options_free(early_config_lp)
            if ec != 0:
                raise self.from_object()(obj)

            config_lp_template = pressio.compressor_get_options(obj)
            config_lp = python_to_pressio_options(compressor_config, config_lp_template)
            ec = pressio.compressor_set_options(obj, config_lp)
            pressio.options_free(config_lp)
            pressio.options_free(config_lp_template)
            if ec != 0:
                raise self.from_object()(obj)

            super().__init__(obj, obj_id, name)

        finally:
            pressio.release(library)

    def __del__(self):
        # works around a bug in swig 4.0 where during shutdown pressio is set to None
        # indicating that the shared library has been unloaded.  Just let the OS
        # clean up in this case.
        if pressio:
            pressio.compressor_release(self._object)

    def get_metrics(self):
        """get runtime metrics"""
        lp_options = pressio.compressor_get_metrics_results(self._object)
        options = pressio_options_to_python(lp_options)
        pressio.options_free(lp_options)
        return options

    def encode(self, buf, out=None):
        """perform compression

        params:
            uncompressed: np.ndarray - the data to be compressed
        """
        uncompressed_lp = None
        compressed_lp = None
        cleanup_comp = True
        try:
            uncompressed_lp = python_to_pressio_data(buf)
            compressed_lp = pressio.data_new_empty(pressio.byte_dtype, pressio.vector_uint64_t()) if out is None else python_to_pressio_data(out)

            rc = pressio.compressor_compress(self._object, uncompressed_lp, compressed_lp)
            if rc:
                raise PressioException.from_compressor(self._object)

            comp, cleanup_comp = pressio_data_to_python(compressed_lp)
        finally:
            pressio.data_free(uncompressed_lp)
            if cleanup_comp:
                pressio.data_free(compressed_lp)
        return comp

    def decode(self, buf, out=None):
        """perform decompression

        params:
            compressed: bytes - the data to be decompressed
            decompressed: numpy.ndarray - memory to be used to decompress data into
        """
        compressed_lp = None
        decompressed_lp = None
        cleanup_dec = True
        try:
            compressed_lp = python_to_pressio_data(buf)
            decompressed_lp = python_to_pressio_data(out)

            rc = pressio.compressor_decompress(self._object, compressed_lp, decompressed_lp)
            if rc:
                raise PressioException.from_compressor(self._object)

            dec, cleanup_dec = pressio_data_to_python(decompressed_lp, out)

            if out is not None:
                return dec
            else:
                return ndarray_copy(dec, out)
        finally:
            pressio.data_free(compressed_lp)
            if cleanup_dec:
                pressio.data_free(decompressed_lp)


class PressioIO(PressioObject):

    @staticmethod
    def _typename():
        return "io"

    def get_name(self):
        return pressio.io_get_name(self._object).decode()

    def from_object(self):
        return PressioException.from_io

    def _get_configuration_impl(self):
        return pressio.io_get_configuration(self._object)

    def _get_options_impl(self):
        return pressio.io_get_options(self._object)

    def _get_documentation_impl(self):
        return pressio.io_get_documentation(self._object)

    def _set_options_impl(self, options):
        return pressio.io_set_options(self._object, options)

    def __init__(self, io, early_config, io_config, name):
        library = None
        config_lp = None
        config_lp_template = None
        try:
            config_lp = None
            library = pressio.instance()
            obj = pressio.get_io(library, io.encode())
            obj_id = io
            if not obj:
                raise PressioException.from_library(library)

            if name is not None:
                pressio.compressor_set_name(obj, name.encode())

            early_config_lp = python_to_pressio_options(early_config)
            pressio.io_set_options(obj, early_config_lp)
            pressio.options_free(early_config_lp)

            config_lp_template = pressio.io_get_options(obj)
            config_lp = python_to_pressio_options(io_config, config_lp_template)

            pressio.io_set_options(obj, config_lp)
            super().__init__(obj, obj_id, name)
        finally:
            pressio.release(library)
            pressio.options_free(config_lp)
            pressio.options_free(config_lp_template)

    def __del__(self):
        # works around a bug in swig 4.0 where during shutdown pressio is set to None
        # indicating that the shared library has been unloaded.  Just let the OS
        # clean up in this case.
        if pressio:
            pressio.io_free(self._object)

    def read(self, template=None):
        """reads a data buffer from a file

        params:
            template: Optional[np.ndarray] - a input template if one is provided
        """
        if template is not None:
            template = python_to_pressio_data(template)
        ret_lp = pressio.io_read(self._object, template)
        if not ret_lp:
            raise PressioException.from_io(self._object)
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
        out = None
        try:
            out = python_to_pressio_data(output)
            ret = pressio.io_write(self._object, out)
            if ret:
                raise PressioException.from_io(self._object)
            return ret
        finally:
            pressio.data_free(out)


class PressioMetrics(PressioObject):
    @staticmethod
    def _typename():
        return "metric"

    def get_name(self):
        return pressio.metrics_get_name(self._object).decode()

    def from_object(self):
        return PressioException.from_metric

    def _get_configuration_impl(self):
        return pressio.metrics_get_configuration(self._object)

    def _get_options_impl(self):
        return pressio.metrics_get_options(self._object)

    def _get_documentation_impl(self):
        return pressio.metrics_get_documentation(self._object)

    def _set_options_impl(self, options):
        return pressio.metrics_set_options(self._object, options)

    def __init__(self, ids, early_config, metrics_config, name):
        library = None
        config_lp = None
        config_lp_template = None
        try:
            library = pressio.instance()
            if isinstance(ids, str):
                ids = [ids]
            metrics_ids = pressio.vector_string([i.encode() for i in ids])
            obj = pressio.new_metrics(library, metrics_ids)
            obj_id = "composite"
            if not obj:
                raise PressioException.from_library(library)

            if name is not None:
                pressio.metrics_set_name(obj, name.encode())

            early_config_lp = python_to_pressio_options(early_config)
            pressio.metrics_set_options(obj, early_config_lp)
            pressio.options_free(early_config_lp)

            config_lp_template = pressio.metrics_get_options(obj)
            config_lp = python_to_pressio_options(metrics_config, config_lp_template)

            pressio.metrics_set_options(obj, config_lp)
            super().__init__(obj, obj_id, name)
        finally:
            pressio.release(library)
            pressio.options_free(config_lp)
            pressio.options_free(config_lp_template)

    def __del__(self):
        # works around a bug in swig 4.0 where during shutdown pressio is set to None
        # indicating that the shared library has been unloaded.  Just let the OS
        # clean up in this case.
        if pressio:
            pressio.metrics_free(self._object)

    def evaluate(self, input=None, compressed=None, output=None):
        results_lp = None
        input_lp = None
        compressed_lp = None
        output_lp = None
        try:
            input_lp = input if input is None else python_to_pressio_data(input)
            compressed_lp = compressed if compressed is None else pressio.io_data_from_bytes(compressed)
            output_lp = output if output is None else python_to_pressio_data(output)
            results_lp = pressio.metrics_evaluate(self._object, input_lp, compressed_lp, output_lp)
            return pressio_options_to_python(results_lp)
        finally:
            if input_lp is not None:
                pressio.data_free(input_lp)
            if output_lp is not None:
                pressio.data_free(output_lp)
            if compressed_lp is not None:
                pressio.data_free(compressed_lp)
            if results_lp is not None:
                pressio.options_free(results_lp)
