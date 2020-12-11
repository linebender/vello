// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

layout(set = 0, binding = 0) buffer Memory {
    // offset into memory of the next allocation, initialized by the user.
    uint mem_offset;
    bool mem_overflow;
    uint[] memory;
};

// Alloc represents a memory allocation.
struct Alloc {
    // offset in bytes into memory.
    uint offset;
    // failed is true if the allocation overflowed memory.
    bool failed;
};

// malloc allocates size bytes of memory.
Alloc malloc(uint size) {
    Alloc a;
	// Round up to nearest 32-bit word.
	size = (size + 3) & ~3;
    a.offset = atomicAdd(mem_offset, size);
    a.failed = a.offset + size > memory.length() * 4;
    if (a.failed) {
        mem_overflow = true;
    }
    return a;
}
