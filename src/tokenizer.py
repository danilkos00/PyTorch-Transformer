from typing import BinaryIO
import os
import regex as re
import multiprocessing
import numpy as np
from tqdm import tqdm


class Tokenizer():
    def __init__(self, vocab=None, merges=None, special_tokens=[], pretokenize_pattern=rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", total_freqs=None):
        if vocab is None:
            self.vocab = {i: bytes([i]) for i in range(256)}
        else:
            self.vocab = vocab

        self.byte_to_id = {v: k for k, v in self.vocab.items()}

        self.special_tokens = {}

        if special_tokens:
            next_id = len(self.vocab)
            for sp_token in special_tokens:
                sp_token_bytes = sp_token.encode("utf-8")
                if sp_token_bytes not in self.byte_to_id:
                    self.vocab[next_id] = sp_token_bytes
                    self.byte_to_id[sp_token_bytes] = next_id
                    self.special_tokens[sp_token_bytes] = next_id
                    next_id += 1
                else:
                    self.special_tokens[sp_token_bytes] = self.byte_to_id[sp_token_bytes]

        if merges is None:
            self.merges = []
        else:
            self.merges = merges

        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

        self.compiled_pattern = re.compile(pretokenize_pattern)

        self.total_freqs = total_freqs


    def _find_chunk_boundaries(
        self,
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), (
            "Must represent special token as a bytestring"
        )

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))


    def _get_freqs_by_chunk(self, args):
        start, end, data_path = args
        separator = re.escape(b'<|endoftext|>')

        with open(data_path, 'rb') as f:
            f.seek(start)
            chunk = f.read(end - start)

        docs = re.splititer(separator, chunk)
        freqs = {}

        for doc in docs:
            for token in re.finditer(self.compiled_pattern, doc):
                byte_token = tuple(bytes([ch]) for ch in token.group(0))
                freqs[byte_token] = freqs.get(byte_token, 0) + 1

        return freqs


    def _get_total(self, data_path):
        num_processes = multiprocessing.cpu_count()

        with open(data_path, 'rb') as f:
            boundaries = self._find_chunk_boundaries(f, num_processes, b'<|endoftext|>')

        tasks = [(start, end, data_path) for start, end in zip(boundaries, boundaries[1:])]

        with multiprocessing.Pool(num_processes) as pool:
            results = pool.map(self._get_freqs_by_chunk, tasks)

        total_freqs = {}

        for dictionary in results:
            for k, v in dictionary.items():
                total_freqs[k] = total_freqs.get(k, 0) + v

        return total_freqs


    def _get_stats(self, word_freqs):
        stats = {}
        pair_to_words = {}
        for word, freq in word_freqs.items():
            for pair in zip(word, word[1:]):
                stats[pair] = stats.get(pair, 0) + freq
                pair_to_words.setdefault(pair, {})[(word, freq)] = pair_to_words.setdefault(pair, {}).get((word, freq), 0) + 1

        return stats, pair_to_words


    def _merge_pair(self, word_freqs, top_pair, stats, pair_to_words):
        affected_words = list(pair_to_words[top_pair])
        for word, freq in affected_words:
            if word not in word_freqs:
                continue

            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and top_pair == (word[i], word[i + 1]):
                    new_word.append(top_pair[0] + top_pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            new_word = tuple(new_word)

            word_freqs[new_word] = word_freqs.pop(word)

            old_pairs = zip(word, word[1:])

            for pair in old_pairs:
                stats[pair] -= freq
                if stats[pair] <= 0:
                    del stats[pair]
                if (word, freq) in pair_to_words[pair]:
                    pair_to_words[pair][(word, freq)] -= 1
                    if pair_to_words[pair][(word, freq)] <= 0:
                        pair_to_words[pair].pop((word, freq))
                    if not pair_to_words[pair]:
                        del pair_to_words[pair]

            new_pairs = zip(new_word, new_word[1:])

            for pair in new_pairs:
                stats[pair] = stats.get(pair, 0) + freq
                pair_to_words.setdefault(pair, {})[(new_word, freq)] = pair_to_words.setdefault(pair, {}).get((new_word, freq), 0) + 1


    def train(self, input_path, vocab_size, verbose=False):
        if self.total_freqs is None:
            self.total_freqs = self._get_total(input_path)

        start_vocab_len = len(self.vocab)
        num_merges = vocab_size - start_vocab_len

        stats, pair_to_words = self._get_stats(self.total_freqs)

        for i in range(num_merges):
            if not stats:
                break

            top_pair = max(stats, key=lambda x: (stats[x], x))

            idx = start_vocab_len + i

            self._merge_pair(self.total_freqs, top_pair, stats, pair_to_words)

            self.vocab[idx] = top_pair[0] + top_pair[1]

            self.merges.append(top_pair)

            if verbose:
                print(f'merge {i + 1}/{num_merges}; {top_pair[0]} was merged with {top_pair[1]}')

        self.byte_to_id = {v: k for k, v in self.vocab.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}


    def _get_bpe_tokens(self, token: bytes) -> list[bytes]:
        word = [bytes([b]) for b in token]
        while True:
            pairs = [(word[i], word[i+1]) for i in range(len(word) - 1)]
            ranked_pairs = [(self.merge_ranks.get(pair, float('inf')), i, pair)
                            for i, pair in enumerate(pairs)]
            if not ranked_pairs:
                break
            ranked_pairs.sort()
            rank, i, pair = ranked_pairs[0]
            if rank == float('inf'):
                break

            word = word[:i] + [pair[0] + pair[1]] + word[i+2:]
        return word


    def encode(self, text: str | bytes) -> list[int]:
        if isinstance(text, str):
            text = text.encode('utf-8')
        elif not isinstance(text, bytes):
            raise ValueError(f'input must be string or bytes, got: {type(text)}')
        tokens = []

        special_pattern = "(" + "|".join(re.escape(k.decode()) for k in self.special_tokens) + ")"
        special_chunks = re.split(special_pattern.encode(), text)

        for chunk in special_chunks:
            if chunk in self.special_tokens:
                tokens.append(self.special_tokens[chunk])
                continue

            for word in re.finditer(self.compiled_pattern, chunk):
                bpe_tokens = self._get_bpe_tokens(word.group(0))
                for token in bpe_tokens:
                    token_id = self.byte_to_id.get(token)
                    if token_id is None:
                        raise ValueError(f"Unknown token {token}")
                    tokens.append(token_id)

        return tokens


    def encode_iterable(self, iterable):
        for line in iterable:
            for token_id in self.encode(line):
                yield token_id


    def decode(self, ids: list[int]) -> str:
        byte_seq = b''.join([self.vocab.get(i, b'') for i in ids])

        return byte_seq.decode('utf-8', errors='replace')


    def _encode_chunk(self, args):
        start, end, path, output_dtype = args

        with open(path, 'rb') as f:
            f.seek(start)
            chunk = f.read(end - start)

        return np.array(self.encode(chunk), dtype=output_dtype)


    def encode_file_by_chunks(self, file_path: str, num_chunks: int, num_processes: int = 1, output_dtype=np.uint16) -> np.ndarray:
        assert num_chunks % num_processes == 0, "num_chunks must be a multiple of the num_processes"

        with open(file_path, 'rb') as f:
            boundaries = self._find_chunk_boundaries(f, num_chunks, b'<|endoftext|>')

        tasks = [(start, end, file_path, output_dtype) for start, end in zip(boundaries, boundaries[1:])]

        ids = []
        for i in tqdm(range(0, len(tasks), num_processes)):
            with multiprocessing.Pool(num_processes) as pool:
                results = pool.map(self._encode_chunk, tasks[i:i+num_processes])
            ids.extend(results)

        return np.concatenate(ids)