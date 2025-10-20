from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Set, Optional, Protocol, Dict
import hashlib, random, time, math


# Hash primitive (stand-in for LiteHash)

def H(data: bytes) -> bytes:
    return hashlib.blake2s(data).digest()



# Merkle tree + proofs

@dataclass(frozen=True)
class ProofElem:
    sibling: bytes
    sibling_is_left: bool

class MerkleTree:
    """Binary Merkle tree with (left || right) node hashing."""
    def __init__(self, leaves: Iterable[bytes]):
        leaves = list(leaves)
        if not leaves:
            raise ValueError("Cannot build Merkle tree with zero leaves.")
        # Treat incoming items as raw blocks; hash to leaves
        self.leaves: List[bytes] = [H(x) for x in leaves]
        self.levels: List[List[bytes]] = []
        self._build()

    def _build(self) -> None:
        level = self.leaves[:]
        self.levels.append(level)
        while len(level) > 1:
            nxt: List[bytes] = []
            # Duplicate last if odd
            if len(level) % 2 == 1:
                level = level + [level[-1]]
            for i in range(0, len(level), 2):
                left, right = level[i], level[i+1]
                nxt.append(H(left + right))
            self.levels.append(nxt)
            level = nxt

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    def gen_proof(self, leaf_index: int) -> List[ProofElem]:
        if not (0 <= leaf_index < len(self.leaves)):
            raise IndexError("leaf_index out of range")
        proof: List[ProofElem] = []
        idx = leaf_index
        for _ in range(0, len(self.levels) - 1):
            level = self.levels[_]
            # Sibling index (flip last bit)
            pair_index = idx ^ 1
            # If odd count, last was duplicated
            if pair_index >= len(level):
                pair_index = len(level) - 1
            sibling = level[pair_index]
            sibling_is_left = (pair_index < idx)
            proof.append(ProofElem(sibling=sibling, sibling_is_left=sibling_is_left))
            idx //= 2
        return proof

def merkle_reconstruct(leaf_hash: bytes, path: List[ProofElem]) -> bytes:
    acc = leaf_hash
    for elem in path:
        acc = H(elem.sibling + acc) if elem.sibling_is_left else H(acc + elem.sibling)
    return acc



# Ed25519 signing via cryptography

class SignatureScheme(Protocol):
    def sign(self, sk: bytes, msg: bytes) -> bytes: ...
    def verify(self, pk: bytes, msg: bytes, sig: bytes) -> bool: ...

class Ed25519Cryptography(SignatureScheme):
    def __init__(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey, Ed25519PublicKey
        )
        from cryptography.hazmat.primitives.serialization import (
            Encoding, PrivateFormat, NoEncryption, PublicFormat
        )
        self._Ed25519PrivateKey = Ed25519PrivateKey
        self._Ed25519PublicKey = Ed25519PublicKey
        self._Encoding = Encoding
        self._PrivateFormat = PrivateFormat
        self._NoEncryption = NoEncryption
        self._PublicFormat = PublicFormat

    def generate_keypair(self) -> Tuple[bytes, bytes]:
        sk_obj = self._Ed25519PrivateKey.generate()
        pk_obj = sk_obj.public_key()
        sk = sk_obj.private_bytes(
            encoding=self._Encoding.Raw,
            format=self._PrivateFormat.Raw,
            encryption_algorithm=self._NoEncryption()
        )
        pk = pk_obj.public_bytes(
            encoding=self._Encoding.Raw,
            format=self._PublicFormat.Raw
        )
        return sk, pk

    def sign(self, sk: bytes, msg: bytes) -> bytes:
        sk_obj = self._Ed25519PrivateKey.from_private_bytes(sk)
        return sk_obj.sign(msg)

    def verify(self, pk: bytes, msg: bytes, sig: bytes) -> bool:
        pk_obj = self._Ed25519PublicKey.from_public_bytes(pk)
        try:
            pk_obj.verify(sig, msg)
            return True
        except Exception:
            return False



# Message binding for (R || e || t_e)

def encode_root_message(R: bytes, e: int, t_e: int) -> bytes:
    # Simple unambiguous encoding + hash; keeps message short
    return H(b"R|" + R + b"|e|" + str(e).encode() + b"|t|" + str(t_e).encode())



# Edge commitment + signature

@dataclass
class EdgeCommitment:
    root: bytes
    epoch: int
    t_epoch: int
    sig: bytes
    pk: bytes

def edge_commit_and_sign(
    blocks: List[bytes],
    epoch: int,
    sigscheme: SignatureScheme,
    sk: bytes,
    pk: bytes,
    t_epoch: Optional[int] = None
) -> Tuple[MerkleTree, EdgeCommitment]:
    t_epoch = int(time.time()) if t_epoch is None else t_epoch
    tree = MerkleTree(leaves=blocks)  # hashes internally
    R = tree.root
    msg = encode_root_message(R, epoch, t_epoch)
    sig = sigscheme.sign(sk, msg)
    return tree, EdgeCommitment(root=R, epoch=epoch, t_epoch=t_epoch, sig=sig, pk=pk)



# Verifier: 

def localize_corruption_for_replica(
    blocks: List[bytes],
    get_proof,  # index -> List[ProofElem]
    commitment: EdgeCommitment,
    sigscheme: SignatureScheme
) -> Tuple[Set[int], Dict[str, float]]:
    """
    Returns:
      - I_i: set of detected corrupted block indices
      - stats: timing stats with keys:
          'blocks' (int), 'total_verify_ns' (int),
          'avg_per_block_ns' (float), 'avg_per_block_ms' (float)
    """
    I_i: Set[int] = set()

    # Signature confirmation (global)
    
    msg = encode_root_message(commitment.root, commitment.epoch, commitment.t_epoch)
    if not sigscheme.verify(commitment.pk, msg, commitment.sig):
        # No authentic root => mark all
        n = len(blocks)
        stats = {
            "blocks": n,
            "total_verify_ns": 0,
            "avg_per_block_ns": 0.0,
            "avg_per_block_ms": 0.0,
        }
        return set(range(n)), stats

    # Per-block localization 
    total_verify_ns = 0
    for j, block in enumerate(blocks):
        t0 = time.perf_counter_ns()

        leaf = H(block)
        path = get_proof(j)
        R_hat = merkle_reconstruct(leaf, path)

        if R_hat != commitment.root:
            I_i.add(j)

        t1 = time.perf_counter_ns()
        total_verify_ns += (t1 - t0)

    n = len(blocks)
    avg_ns = (total_verify_ns / n) if n else 0.0
    stats = {
        "blocks": n,
        "total_verify_ns": int(total_verify_ns),
        "avg_per_block_ns": float(avg_ns),
        "avg_per_block_ms": float(avg_ns / 1e6),
    }
    return I_i, stats



# Real-time-ish multi-node simulation (random selection)  

def simulate_multi_node(
    num_nodes: int = 15,
    blocks_per_node: int = 64,
    block_size_mb: int = 0,                 # per-block payload size in MB (0 -> tiny demo blocks)
    corrupt_node_ids: Optional[List[int]] = None,  # None => choose randomly
    num_corrupt_nodes: int = 10,                    # used when corrupt_node_ids is None
    corruption_fraction: float = 0.25,             # 25% of blocks on corrupted nodes
    seed: int = 42,
    per_node_random_fraction: bool = False         # if True, draw fraction per node in [0,corruption_fraction]
):
    """
    Simulates Stage-2 across num_nodes.
    If corrupt_node_ids is None, randomly selects num_corrupt_nodes distinct nodes to corrupt.

    NOTE: Large block_size_mb values (e.g., 20) multiply memory:
      total_bytes ~= num_nodes * blocks_per_node * block_size_mb * 1024 * 1024
    Adjust num_nodes/blocks_per_node accordingly.
    """
    random.seed(seed)

    # --- Choose nodes to corrupt 
    
    if corrupt_node_ids is None:
        if not (0 <= num_corrupt_nodes <= num_nodes):
            raise ValueError("num_corrupt_nodes must be between 0 and num_nodes.")
        corrupt_node_ids = sorted(random.sample(range(num_nodes), k=num_corrupt_nodes))

    # Setup signature scheme + keys per node
    
    sigscheme = Ed25519Cryptography()
    keys: Dict[int, Tuple[bytes, bytes]] = {}
    for i in range(num_nodes):
        sk, pk = sigscheme.generate_keypair()
        keys[i] = (sk, pk)

    # Create raw blocks per node (seeded for reproducibility)
    
    replicas: Dict[int, List[bytes]] = {}
    for i in range(num_nodes):
        blocks = []
        for j in range(blocks_per_node):
            prefix = f"node={i},block={j},".encode()
            if block_size_mb > 0:
                size = block_size_mb * 1024 * 1024
                # For determinism + speed, use random.randbytes; for crypto use secrets.token_bytes/os.urandom
                payload = prefix + random.randbytes(size)
            else:
                payload = prefix + random.randbytes(24)
            blocks.append(payload)
        replicas[i] = blocks

    # Edge-side commit + sign for each node (epoch can be any int)
    epoch = 100
    trees: Dict[int, MerkleTree] = {}
    commits: Dict[int, EdgeCommitment] = {}
    for i in range(num_nodes):
        sk, pk = keys[i]
        tree, commit = edge_commit_and_sign(
            replicas[i], epoch, sigscheme, sk, pk, t_epoch=None
        )
        trees[i] = tree
        commits[i] = commit

    # Tamper (AFTER signing) on selected nodes
    
    tampered: Dict[int, List[bytes]] = {i: list(replicas[i]) for i in range(num_nodes)}
    corruption_map: Dict[int, Set[int]] = {i: set() for i in range(num_nodes)}

    for nid in corrupt_node_ids:
        frac = (random.random() * corruption_fraction) if per_node_random_fraction else corruption_fraction
        # Use ceil so small block counts can still yield >1 corrupted block; cap at blocks_per_node.
        num_to_corrupt = max(2, min(blocks_per_node, math.ceil(blocks_per_node * frac)))  # increase corrupted per replica
        bad_indices = sorted(random.sample(range(blocks_per_node), num_to_corrupt))
        for j in bad_indices:
            # Flip/alter the block deterministically under seed
            tampered[nid][j] = tampered[nid][j] + b"|CORRUPTED|" + random.randbytes(8)
        corruption_map[nid] = set(bad_indices)

    # Verifier: run localization for all nodes 
    
    detected: Dict[int, Set[int]] = {}
    timing_stats: Dict[int, Dict[str, float]] = {}
    for i in range(num_nodes):
        I_i, stats = localize_corruption_for_replica(
            blocks=tampered[i],
            get_proof=trees[i].gen_proof,
            commitment=commits[i],
            sigscheme=sigscheme
        )
        detected[i] = I_i
        timing_stats[i] = stats

    # Results
    
    print("=== Simulation Summary ===")
    print(f"Nodes       : {num_nodes}")
    print(f"Blocks/node : {blocks_per_node}")
    print(f"Block size  : {block_size_mb} MB per block" if block_size_mb > 0 else "Block size  : ~40–50 bytes (demo)")
    print(f"Corrupted nodes (selected) : {corrupt_node_ids}")
    if per_node_random_fraction:
        print(f"Corruption fraction per node: random in [0, {corruption_fraction:.2f}]")
    else:
        print(f"Corruption fraction (fixed) : {corruption_fraction:.2f}")
    print()

    # Timing summary 
    
    total_blocks = sum(stats["blocks"] for stats in timing_stats.values())
    total_ns = sum(stats["total_verify_ns"] for stats in timing_stats.values())
    overall_avg_ms = (total_ns / total_blocks) / 1e6 if total_blocks else 0.0
    print("=== Timing (localization/verification) ===")
    for i in range(num_nodes):
        s = timing_stats[i]
        print(f"Node {i}: avg time per block = {s['avg_per_block_ms']:.3f} ms "
              f"(blocks={s['blocks']}, total={s['total_verify_ns']/1e9:.3f} s)")
    print(f"\nOverall average time per block = {overall_avg_ms:.3f} ms "
          f"(across {total_blocks} blocks)\n")

    total_true = sum(len(corruption_map[i]) for i in range(num_nodes))
    total_detected = sum(len(detected[i]) for i in range(num_nodes))
    print(f"Total true corrupted blocks   : {total_true}")
    print(f"Total detected corrupted      : {total_detected}")
    print()

    for i in range(num_nodes):
        truth = corruption_map[i]
        guess = detected[i]
        tp = len(truth & guess)
        fp = len(guess - truth)
        fn = len(truth - guess)
        note = ""
        if tp or fp or fn:
            note = f"(TP={tp}, FP={fp}, FN={fn})"
        print(f"Node {i}:")
        print(f"  True corrupted indices    : {sorted(truth) if truth else '—'}")
        print(f"  Detected corrupted indices: {sorted(guess) if guess else '—'} {note}")
        print()

if __name__ == "__main__":
    simulate_multi_node(
        num_nodes=15,
        blocks_per_node=10,
        block_size_mb=20,      # <<< 20 MB per block (large memory usage)
        corrupt_node_ids=None, # random selection
        num_corrupt_nodes=10,
        corruption_fraction=0.30,
        seed=1337,
        per_node_random_fraction=False
    )

