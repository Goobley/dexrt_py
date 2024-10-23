import numpy as np


# https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
# // "Insert" a 0 bit after each of the 16 low bits of x
def part_1_by_1(x: np.uint32):
    x &= 0x0000FFFF  # x = ---- ---- ---- ---- fedc ba98 7654 3210
    x = (x ^ (x << 8)) & 0x00FF00FF  # x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x << 4)) & 0x0F0F0F0F  # x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x << 2)) & 0x33333333  # x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x << 1)) & 0x55555555  # x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    return x


# Inverse of Part1By1 - "delete" all odd-indexed bits
def compact_1_by_1(x: np.uint32):
    x &= 0x55555555  # x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    x = (x ^ (x >> 1)) & 0x33333333  # x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x >> 2)) & 0x0F0F0F0F  # x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x >> 4)) & 0x00FF00FF  # x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x >> 8)) & 0x0000FFFF  # x = ---- ---- ---- ---- fedc ba98 7654 3210
    return x


def encode_morton_2(x: np.int32, z: np.int32):
    return (part_1_by_1(np.uint32(z)) << 1) + part_1_by_1(np.uint32(x))


def decode_morton_2(code: np.uint32):
    return compact_1_by_1(code >> 0), compact_1_by_1(code >> 1)  # x, z
