import heapq
import os
from collections import defaultdict, Counter


class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(freq_map):
    priority_queue = [HuffmanNode(char, freq) for char, freq in freq_map.items()]
    heapq.heapify(priority_queue)

    while len(priority_queue) > 1:
        left_child = heapq.heappop(priority_queue)
        right_child = heapq.heappop(priority_queue)
        internal_node = HuffmanNode(None, left_child.freq + right_child.freq)
        internal_node.left = left_child
        internal_node.right = right_child
        heapq.heappush(priority_queue, internal_node)

    return priority_queue[0]


def build_huffman_codes(node, current_code, huffman_codes):
    if node is None:
        return

    if node.char is not None:
        huffman_codes[node.char] = current_code
        return

    build_huffman_codes(node.left, current_code + '0', huffman_codes)
    build_huffman_codes(node.right, current_code + '1', huffman_codes)


def compress(input_file, output_file):
    with open(input_file, 'rb') as file:
        content = file.read()

    freq_map = Counter(content)
    root = build_huffman_tree(freq_map)
    huffman_codes = {}
    build_huffman_codes(root, '', huffman_codes)

    encoded_content = ''.join(huffman_codes[char] for char in content)

    padding = 8 - len(encoded_content) % 8
    encoded_content = f"{encoded_content:0>{len(encoded_content) + padding}}"

    encoded_bytes = bytearray([int(encoded_content[i:i+8], 2) for i in range(0, len(encoded_content), 8)])

    with open(output_file, 'wb') as file:
        file.write(bytes([padding])) 
        file.write(bytes(huffman_codes)) 
        file.write(bytes(encoded_bytes))  


def decompress(input_file, output_file):
    with open(input_file, 'rb') as file:
        padding = int.from_bytes(file.read(1), byteorder='big')
        huffman_codes_bytes = file.read(256 * 2) 
        

        huffman_codes = {}
        for i in range(0, len(huffman_codes_bytes), 2):
            char = huffman_codes_bytes[i:i + 1]
            code = huffman_codes_bytes[i + 1:i + 2]
            huffman_codes[code] = char

        encoded_content = ''
        byte = file.read(1)
        while byte:
            encoded_content += format(ord(byte), '08b')
            byte = file.read(1)

    encoded_content = encoded_content[:-padding] 

    decoded_content = ''
    current_code = ''
    for bit in encoded_content:
        current_code += bit
        if current_code in huffman_codes:
            decoded_char = huffman_codes[current_code].decode('utf-8')
            decoded_content += decoded_char
            current_code = ''

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(decoded_content)
        
compress('dct.avi', 'compressed_file_az.bin')
# decompress('compressed_file.bin', 'output_binary_file.mp4')
