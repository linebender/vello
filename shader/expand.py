# Expand an 8 bit mask to 8x 4 bits
def expand(m):
    c = m
    c |= m << 6
    c |= c << 12
    d = c << 3
    return (c & 0x1010101) | (d & 0x10101010)

for i in (0, 1, 2, 4, 8, 16, 32, 64, 128, 3, 255):
    print(hex(i), hex(expand(i)))
