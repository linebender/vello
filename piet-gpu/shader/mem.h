// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

layout(set = 0, binding = 0) buffer Memory {
    // offset into memory of the next allocation, initialized by the user.
    uint mem_offset;
    // mem_error tracks the status of memory accesses, initialized to NO_ERROR
    // by the user. ERR_MALLOC_FAILED is reported for insufficient memory.
    // If MEM_DEBUG is defined the following errors are reported:
    // - ERR_OUT_OF_BOUNDS is reported for out of bounds writes.
    // - ERR_UNALIGNED_ACCESS for memory access not aligned to 32-bit words.
    uint mem_error;
    uint[] memory;
};

// Uncomment this line to add the size field to Alloc and enable memory checks.
// Note that the Config struct in setup.h grows size fields as well.
//#define MEM_DEBUG

#define NO_ERROR 0
#define ERR_MALLOC_FAILED 1
#define ERR_OUT_OF_BOUNDS 2
#define ERR_UNALIGNED_ACCESS 3

#define Alloc_size 8

// Alloc represents a memory allocation.
struct Alloc {
    // offset in bytes into memory.
    uint offset;
#ifdef MEM_DEBUG
    // size in bytes of the allocation.
    uint size;
#endif
};

struct MallocResult {
    Alloc alloc;
    // failed is true if the allocation overflowed memory.
    bool failed;
};

// new_alloc synthesizes an Alloc when its offset and size are derived.
Alloc new_alloc(uint offset, uint size) {
    Alloc a;
    a.offset = offset;
#ifdef MEM_DEBUG
    a.size = size;
#endif
    return a;
}

// malloc allocates size bytes of memory.
MallocResult malloc(uint size) {
    MallocResult r;
    r.failed = false;
    uint offset = atomicAdd(mem_offset, size);
    r.alloc = new_alloc(offset, size);
    if (offset + size > memory.length() * 4) {
        r.failed = true;
        atomicMax(mem_error, ERR_MALLOC_FAILED);
        return r;
    }
#ifdef MEM_DEBUG
    if ((size & 3) != 0) {
        r.failed = true;
        atomicMax(mem_error, ERR_UNALIGNED_ACCESS);
        return r;
    }
#endif
    return r;
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
#endif
    return new_alloc(a.offset + offset, size);
}
