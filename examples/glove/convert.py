#!/usr/bin/env python3

import argparse
import sys
import struct
import numpy as np
import tqdm

def process_glove_file(input_file, output_file, dim):
    matrix = []
    # Compute nuber of lines for progress bar
    num_lines = sum(1 for line in open(input_file, 'r'))
    with open(input_file, 'r') as inf:
        with open(output_file, 'wb') as ouf:
            for line in tqdm.tqdm(inf, total=num_lines):
                row = [float(x) for x in line.split()[-dim:]]
                ouf.write(struct.pack('i', len(row)))
                ouf.write(struct.pack('%sf' % len(row), *row))
                matrix.append(np.array(row, dtype=np.float32))

    np.save(output_file, np.array(matrix))

def main():
    parser = argparse.ArgumentParser(description="Process GloVe text files")
    parser.add_argument("dim", type=int, help="Dimension of GloVe vectors",)
    parser.add_argument("input_file", metavar="input", type=str,
                        help="GloVe input file in text format",)
    args = parser.parse_args()

    output_file = args.input_file.replace(".txt", "")
    print(f"Processing {args.input_file}...")
    process_glove_file(args.input_file, output_file, args.dim)
    print(f"Completed processing {args.input_file}")
    print(f"Output saved to {output_file}.npy")

if __name__ == "__main__":
    main()

