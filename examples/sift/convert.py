#!/usr/bin/env python3

import argparse
import struct
import numpy as np
import tqdm
import tarfile


def load_texmex_vectors(file, num_vectors, vector_dim):
    vectors = np.zeros((num_vectors, vector_dim))
    for i in tqdm.tqdm(range(num_vectors)):
        file.read(4)  # ignore vector length
        vectors[i] = struct.unpack("f" * vector_dim, file.read(vector_dim * 4))
    return vectors


def get_irisa_matrix(tar, filename):
    member = tar.getmember(filename)
    file = tar.extractfile(member)
    vector_dim, = struct.unpack("i", file.read(4))
    num_vectors = member.size // (4 + 4 * vector_dim)
    file.seek(0)
    return load_texmex_vectors(file, num_vectors, vector_dim)


def process_glove_file(input_file, output_file, dim):
    matrix = []
    # Compute nuber of lines for progress bar
    num_lines = sum(1 for line in open(input_file, "r"))
    with open(input_file, "r") as inf:
        with open(output_file, "wb") as ouf:
            for line in tqdm.tqdm(inf, total=num_lines):
                row = [float(x) for x in line.split()[-dim:]]
                ouf.write(struct.pack("i", len(row)))
                ouf.write(struct.pack("%sf" % len(row), *row))
                matrix.append(np.array(row, dtype=np.float32))

    np.save(output_file, np.array(matrix))


def main():
    parser = argparse.ArgumentParser(description="Process Sift")
    parser.add_argument(
        "sift_file",
        type=str,
        help="sift.tar.gz file",
    )
    args = parser.parse_args()

    print(f"Processing {args.sift_file}...")
    with tarfile.open(args.sift_file, "r:gz") as t:
        train = get_irisa_matrix(t, "sift/sift_base.fvecs")
        test = get_irisa_matrix(t, "sift/sift_query.fvecs")

    output_file = args.sift_file.replace(".tar.gz", "")
    print(f"Saving to {output_file}.npy")
    np.save(output_file, np.vstack([train, test]))


if __name__ == "__main__":
    main()


def sift(out_fn):
    import tarfile

    url = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
    fn = os.path.join("data", "sift.tar.tz")
    download(url, fn)
    with tarfile.open(fn, "r:gz") as t:
        train = _get_irisa_matrix(t, "sift/sift_base.fvecs")
        test = _get_irisa_matrix(t, "sift/sift_query.fvecs")
        write_output(train, test, out_fn, "euclidean")
