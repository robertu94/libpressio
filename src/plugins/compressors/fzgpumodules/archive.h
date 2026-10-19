#pragma once
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
#include "libpressio_ext/cpp/data.h"
#include "fzm_format.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

// FZLP v1: prefix, dimensions, zero padding to 16 bytes, FZM metadata, payload.
// All integers are little endian; dtype codes are independent of libpressio enums.
struct ArchiveMetadata {
    static constexpr size_t prefix_size = 40;
    static constexpr uint64_t magic = 0x31303030504c5a46ULL; // FZLP0001
    pressio_dtype dtype;
    std::vector<size_t> dims;
    std::vector<uint8_t> header;
    size_t payload_offset;
    size_t payload_size;
    size_t output_size;
    std::vector<uint64_t> segment_sizes;
};

inline uint64_t archive_read(const uint8_t* p, size_t n) {
    uint64_t value = 0;
    for(size_t i = 0; i < n; ++i) value |= uint64_t(p[i]) << (8*i);
    return value;
}
inline void archive_write(std::vector<uint8_t>& out, uint64_t value, size_t n) {
    for(size_t i = 0; i < n; ++i) out.push_back(uint8_t(value >> (8*i)));
}
inline const std::vector<pressio_dtype>& archive_dtypes() {
    static const std::vector<pressio_dtype> types = {
        pressio_float_dtype, pressio_double_dtype,
        pressio_int8_dtype, pressio_int16_dtype, pressio_int32_dtype, pressio_int64_dtype,
        pressio_uint8_dtype, pressio_uint16_dtype, pressio_uint32_dtype, pressio_uint64_dtype};
    return types;
}
inline size_t archive_metadata_size(const uint8_t* prefix, size_t total_size) {
    if(total_size < ArchiveMetadata::prefix_size ||
       archive_read(prefix, 8) != ArchiveMetadata::magic || archive_read(prefix+8, 4) != 1)
        throw std::runtime_error("Invalid FZGM archive magic/version; legacy raw buffers are not self-contained");
    const auto rank = archive_read(prefix+16, 4);
    const auto header_size = archive_read(prefix+24, 8);
    const auto payload_size = archive_read(prefix+32, 8);
    if(rank == 0 || rank > 32 || archive_read(prefix+20, 4) != 0)
        throw std::runtime_error("Invalid FZGM archive dimensions/flags");
    const size_t base = (ArchiveMetadata::prefix_size + size_t(rank)*8 + 15) & ~size_t(15);
    if(base > total_size || header_size > total_size-base ||
       payload_size != total_size-base-header_size ||
       header_size < sizeof(fz::FZMHeaderCore) ||
       header_size > sizeof(fz::FZMHeaderCore) + fz::FZM_MAX_BUFFERS *
                     (sizeof(fz::FZMStageInfo) + sizeof(fz::FZMBufferEntry)))
        throw std::runtime_error("Truncated or inconsistent FZGM archive lengths");
    return base + size_t(header_size);
}
inline ArchiveMetadata parse_archive_metadata(const uint8_t* data, size_t total_size) {
    ArchiveMetadata out;
    out.payload_offset = archive_metadata_size(data, total_size);
    out.payload_size = size_t(archive_read(data+32, 8));
    auto code = archive_read(data+12, 4);
    if(code >= archive_dtypes().size()) throw std::runtime_error("Invalid FZGM archive dtype");
    out.dtype = archive_dtypes()[code];
    out.output_size = size_t(pressio_dtype_size(out.dtype));
    const size_t rank = size_t(archive_read(data+16, 4));
    for(size_t i = 0; i < rank; ++i) {
        const uint64_t dim = archive_read(data+ArchiveMetadata::prefix_size+8*i, 8);
        if(dim == 0 || dim > std::numeric_limits<size_t>::max()/out.output_size)
            throw std::runtime_error("Invalid or overflowing FZGM archive shape");
        out.dims.push_back(size_t(dim));
        out.output_size *= size_t(dim);
    }
    const size_t offset = (ArchiveMetadata::prefix_size+8*rank+15) & ~size_t(15);
    for(size_t i = ArchiveMetadata::prefix_size+8*rank; i < offset; ++i)
        if(data[i] != 0) throw std::runtime_error("Invalid FZGM archive padding");
    out.header.assign(data+offset, data+out.payload_offset);
    fz::FZMHeaderCore core;
    std::memcpy(&core, out.header.data(), sizeof(core));
    if(core.header_size != out.header.size() || core.uncompressed_size != out.output_size ||
       core.compressed_size > total_size ||
       core.num_stages > fz::FZM_MAX_BUFFERS || core.num_buffers == 0 ||
       core.num_buffers > fz::FZM_MAX_BUFFERS ||
       out.header.size() != sizeof(core) + size_t(core.num_stages)*sizeof(fz::FZMStageInfo) +
                            size_t(core.num_buffers)*sizeof(fz::FZMBufferEntry))
        throw std::runtime_error("FZGM archive shape/header disagrees with FZM metadata");
    const size_t entries = sizeof(core) + size_t(core.num_stages)*sizeof(fz::FZMStageInfo);
    uint64_t extent = 0;
    for(size_t i = 0; i < core.num_buffers; ++i) {
        fz::FZMBufferEntry entry;
        std::memcpy(&entry, out.header.data()+entries+i*sizeof(entry), sizeof(entry));
        if(entry.byte_offset != extent || entry.data_size > core.compressed_size-std::min(extent,core.compressed_size) ||
           entry.data_size > std::numeric_limits<uint64_t>::max()-15)
            throw std::runtime_error("Invalid FZGM archive segment extent");
        out.segment_sizes.push_back(entry.data_size);
        const auto slot = (entry.data_size+15) & ~uint64_t(15);
        if(slot > core.compressed_size-std::min(extent,core.compressed_size))
            throw std::runtime_error("Invalid FZGM archive padded segment extent");
        extent += slot;
    }
    if(extent != core.compressed_size)
        throw std::runtime_error("Invalid FZGM archive FZM payload length");
    // FZM file segments exclude the raw multi-output concat header and pad single outputs.
    const uint64_t raw_size = core.num_buffers == 1 ? out.segment_sizes[0]
        : extent + ((4 + uint64_t(core.num_buffers)*8 + 15) & ~uint64_t(15));
    if(raw_size != out.payload_size)
        throw std::runtime_error("FZGM archive payload length disagrees with segment metadata");
    return out;
}
inline std::vector<uint8_t> make_archive_metadata(const pressio_data& input,
                                                  const std::vector<uint8_t>& header,
                                                  size_t payload_size) {
    const auto& types = archive_dtypes();
    auto type = std::find(types.begin(), types.end(), input.dtype());
    auto dims = input.dimensions();
    if(type == types.end() || dims.empty() || dims.size() > 32)
        throw std::runtime_error("Unsupported FZGM archive dtype/rank");
    std::vector<uint8_t> out;
    archive_write(out, ArchiveMetadata::magic, 8);
    archive_write(out, 1, 4);
    archive_write(out, size_t(type-types.begin()), 4);
    archive_write(out, dims.size(), 4);
    archive_write(out, 0, 4);
    archive_write(out, header.size(), 8);
    archive_write(out, payload_size, 8);
    for(size_t dim : dims) archive_write(out, dim, 8);
    while(out.size() % 16) out.push_back(0);
    out.insert(out.end(), header.begin(), header.end());
    return out;
}

}}}
