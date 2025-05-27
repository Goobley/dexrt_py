import numpy as np


# https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
# // "Insert" a 0 bit after each of the 16 low bits of x
def part_1_by_1(x: np.uint32):
    x &= 0x0000FFFF                  # x = ---- ---- ---- ---- fedc ba98 7654 3210
    x = (x ^ (x << 8)) & 0x00FF00FF  # x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x << 4)) & 0x0F0F0F0F  # x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x << 2)) & 0x33333333  # x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x << 1)) & 0x55555555  # x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    return x

# "Insert" two 0 bits after each of the 10 low bits of x
def part_1_by_2(x: np.uint32):
  x &= 0x000003ff                  # x = ---- ---- ---- ---- ---- --98 7654 3210
  x = (x ^ (x << 16)) & 0xff0000ff # x = ---- --98 ---- ---- ---- ---- 7654 3210
  x = (x ^ (x <<  8)) & 0x0300f00f # x = ---- --98 ---- ---- 7654 ---- ---- 3210
  x = (x ^ (x <<  4)) & 0x030c30c3 # x = ---- --98 ---- 76-- --54 ---- 32-- --10
  x = (x ^ (x <<  2)) & 0x09249249 # x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
  return x

# Inverse of Part1By1 - "delete" all odd-indexed bits
def compact_1_by_1(x: np.uint32):
    x &= 0x55555555                  # x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    x = (x ^ (x >> 1)) & 0x33333333  # x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x >> 2)) & 0x0F0F0F0F  # x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x >> 4)) & 0x00FF00FF  # x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x >> 8)) & 0x0000FFFF  # x = ---- ---- ---- ---- fedc ba98 7654 3210
    return x

def compact_1_by_2(x: np.uint32):
  x &= 0x09249249                  # x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
  x = (x ^ (x >>  2)) & 0x030c30c3 # x = ---- --98 ---- 76-- --54 ---- 32-- --10
  x = (x ^ (x >>  4)) & 0x0300f00f # x = ---- --98 ---- ---- 7654 ---- ---- 3210
  x = (x ^ (x >>  8)) & 0xff0000ff # x = ---- --98 ---- ---- ---- ---- 7654 3210
  x = (x ^ (x >> 16)) & 0x000003ff # x = ---- ---- ---- ---- ---- --98 7654 3210
  return x


def encode_morton_2(x: np.int32, z: np.int32):
    """Encodes the lower 16 bits of two int32 into a uint32"""
    return (part_1_by_1(np.uint32(z)) << 1) + part_1_by_1(np.uint32(x))

def encode_morton_3(x: np.int32, y: np.int32, z: np.int32):
    """Encodes the lower 10 bits of three int32 into a uint32"""
    return (
        (part_1_by_2(np.uint32(z)) << 2)
        + (part_1_by_2(np.uint32(y)) << 1)
        + part_1_by_2(np.uint32(x))
    )

def decode_morton_2(code: np.uint32):
    """Decodes a uint32 morton code
    Returns: x, z as uint32
    """
    return compact_1_by_1(code >> 0), compact_1_by_1(code >> 1)  # x, z

def decode_morton_3(code: np.uint32):
    """Decodes a uint32 3d morton code
    Returns: x, y, z as uint32
    """
    return compact_1_by_2(code >> 0), compact_1_by_2(code >> 1), compact_1_by_2(code >> 2)  # x, y, z
