// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

layout(set = 0, binding = 0) buffer Memory {
    // offset into memory of the next allocation, initialized by the user.
    uint mem_offset;
    // mem_error is a bitmask of stages that have failed allocation.
    uint mem_error;
    // offset into blend memory of allocations for blend stack.
    uint blend_offset;
    uint[] memory;
};

// Uncomment this line to add the size field to Alloc and enable memory checks.
// Note that the Config struct in setup.h grows size fields as well.

// This setting is not working and the mechanism will be removed.
//#define MEM_DEBUG

#ifdef MEM_DEBUG
#define Alloc_size 16
#else
// TODO: this seems wrong
#define Alloc_size 8
#endif

// Alloc represents a memory allocation.
struct Alloc {
    // offset in bytes into memory.
    uint offset;
#ifdef MEM_DEBUG
    // size in bytes of the allocation.
    uint size;
#endif
};

// new_alloc synthesizes an Alloc from an offset and size.
Alloc new_alloc(uint offset, uint size, bool mem_ok) {
    Alloc a;
    a.offset = offset;
#ifdef MEM_DEBUG
    if (mem_ok) {
        a.size = size;
    } else {
        a.size = 0;
    }
#endif
    return a;
}

#define STAGE_BINNING (1u << 0)
#define STAGE_TILE_ALLOC (1u << 1)
#define STAGE_PATH_COARSE (1u << 2)
#define STAGE_COARSE (1u << 3)

// Allocations in main memory will never be 0, and this might be slightly
// faster to test against than some other value.
#define MALLOC_FAILED 0

// Check that previous dependent stages have succeeded.
bool check_deps(uint dep_stage) {
    // TODO: this should be an atomic relaxed load, but that involves
    // bringing in "memory scope semantics"
    return (atomicOr(mem_error, 0) & dep_stage) == 0;
}

// Allocate size bytes of memory, offset in bytes.
// Note: with a bit of rearrangement of header files, we could make the
// mem_size argument go away (it comes from the config binding).
uint malloc_stage(uint size, uint mem_size, uint stage) {
    uint offset = atomicAdd(mem_offset, size);
    if (offset + size > mem_size) {
        atomicOr(mem_error, stage);
        offset = MALLOC_FAILED;
    }
    return offset;
}

// touch_mem checks whether access to the memory word at offset is valid.
// If MEM_DEBUG is defined, touch_mem returns false if offset is out of bounds.
// Offset is in words.
bool touch_mem(Alloc alloc, uint offset) {
#ifdef MEM_DEBUG
    if (offset < alloc.offset/4 || offset >= (alloc.offset + alloc.size)/4) {
        atomicMax(mem_error, ERR_OUT_OF_BOUNDS);
        return false;
    }
#endif
    return true;
}

// write_mem writes val to memory at offset.
// Offset is in words.
void write_mem(Alloc alloc, uint offset, uint val) {
    if (!touch_mem(alloc, offset)) {
        return;
    }
    memory[offset] = val;
}

// read_mem reads the value from memory at offset.
// Offset is in words.
uint read_mem(Alloc alloc, uint offset) {
    if (!touch_mem(alloc, offset)) {
        return 0;
    }
    uint v = memory[offset];
    return v;
}

// slice_mem returns a sub-allocation inside another. Offset and size are in
// bytes, relative to a.offset.
Alloc slice_mem(Alloc a, uint offset, uint size) {
#ifdef MEM_DEBUG
    if ((offset & 3) != 0 || (size & 3) != 0) {
        atomicMax(mem_error, ERR_UNALIGNED_ACCESS);
        return Alloc(0, 0);
    }
    if (offset + size > a.size) {
        // slice_mem is sometimes used for slices outside bounds,
        // but never written.
        return Alloc(0, 0);
    }
    return Alloc(a.offset + offset, size);
#else
    return Alloc(a.offset + offset);
#endif
}

// alloc_write writes alloc to memory at offset bytes.
void alloc_write(Alloc a, uint offset, Alloc alloc) {
    write_mem(a, offset >> 2, alloc.offset);
#ifdef MEM_DEBUG
    write_mem(a, (offset >> 2) + 1, alloc.size);
#endif
}

// alloc_read reads an Alloc from memory at offset bytes.
Alloc alloc_read(Alloc a, uint offset) {
    Alloc alloc;
    alloc.offset = read_mem(a, offset >> 2);
#ifdef MEM_DEBUG
    alloc.size = read_mem(a, (offset >> 2) + 1);
#endif
    return alloc;
}
