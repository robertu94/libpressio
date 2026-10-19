"""Python conveniences for the optional FZGPUModules libpressio plugin."""

import libpressio as lp
from libpressio import pressio


class Compressor(lp.PressioCompressor):
    """FZGM compressor with archive-driven decoding and inherited libpressio APIs.

    An empty instance can decode an archive. Encoding requires explicit stages,
    supplied through Chain, from_config(), or set_options().
    """

    def __init__(self, compressor_id="fzgpumodules", early_config=None,
                 compressor_config=None, name=""):
        self._object = None
        if compressor_id != "fzgpumodules":
            raise ValueError("Compressor only supports the fzgpumodules plugin")
        super().__init__(compressor_id, early_config or {}, compressor_config or {}, name)

    def decode(self, buf, out=None):
        """Decode an array or saved bytes; infer output dtype and shape if omitted."""
        compressed_lp = None
        decompressed_lp = None
        cleanup_dec = True
        try:
            compressed_lp = (pressio.io_data_from_bytes(buf) if isinstance(buf, bytes)
                             else lp.python_to_new_pressio_data(buf))
            decompressed_lp = (pressio.data_new_empty(pressio.byte_dtype, pressio.vector_uint64_t())
                               if out is None else lp.python_to_new_pressio_data(out))
            if pressio.compressor_decompress(self._object, compressed_lp, decompressed_lp):
                raise lp.PressioException.from_compressor(self._object)
            if lp.is_pressio_data(buf):
                cleanup_dec = False
                return decompressed_lp
            result, cleanup_dec = lp.pressio_data_to_python(decompressed_lp, out)
            return result
        finally:
            pressio.data_free(compressed_lp)
            if cleanup_dec:
                pressio.data_free(decompressed_lp)


class Chain:
    """Build a linear pipeline for the optional FZGPUModules plugin.

    Creating a compressor requires libpressio built with FZGPUModules support.
    Encoded output includes pipeline metadata, dtype, and dimensions. A fresh
    FZGPUModules compressor can decode it without reconstructing this chain.

    Each add() call appends one stage, auto-numbers it sN, and wires it to
    the previous stage's output (default port, or a named one via
    from_port). Keyword args become that stage's "fzgpumodules:sN:<key>"
    options.

    >>> compressor = (Chain().add("lorenzo:float:uint16", quant_radius=999)
    ...               .add("rze", from_port="codes", chunk_size=8192)
    ...               .configure(eb=1e-3, specialization="auto"))
    """

    def __init__(self):
        self._stages = []
        self._connections = []
        self._options = {}

    def add(self, token, from_port=None, **options):
        sid = f"s{len(self._stages)}"
        if self._stages:
            prev = f"s{len(self._stages) - 1}"
            src = f"{prev}:{from_port}" if from_port else prev
            self._connections.append(f"{sid} <- {src}")
        elif from_port is not None:
            raise ValueError("from_port has no effect on the first stage (nothing precedes it)")
        self._stages.append(token)
        for key, value in options.items():
            self._options[f"fzgpumodules:{sid}:{key}"] = value
        return self

    def _early_config(self):
        cfg = {
            "fzgpumodules:stages": list(self._stages),
            "fzgpumodules:connections": list(self._connections),
        }
        cfg.update(self._options)
        return cfg

    def configure(self, eb=None, metric="size", **top_level_options):
        """Finish the chain and return a compressor; eb is an absolute error bound.

        Remaining keywords are fzgpumodules options, e.g. specialization="auto".
        Omitting eb retains the plugin's default absolute bound of 1e-3.
        """
        early_config = self._early_config()
        early_config.update({f"fzgpumodules:{key}": value
                             for key, value in top_level_options.items()})
        compressor_config = {"pressio:metric": metric}
        if eb is not None:
            compressor_config["pressio:abs"] = eb
        return Compressor(early_config=early_config, compressor_config=compressor_config)
