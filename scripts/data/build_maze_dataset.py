from typing import Optional, Tuple
import os
import csv
import json
import numpy as np

import click
from tqdm import tqdm
from huggingface_hub import hf_hub_download


CHARSET = "# SGo"  # wall, space, start, goal, path(solution)


def dihedral_transform(arr: np.ndarray, idx: int) -> np.ndarray:
    """Apply one of 8 dihedral group transformations (D4).
    
    idx 0-3: rotations (0°, 90°, 180°, 270°)
    idx 4-7: reflections (horizontal, vertical, diagonal, anti-diagonal)
    """
    if idx == 0:
        return arr
    elif idx == 1:
        return np.rot90(arr, 1)
    elif idx == 2:
        return np.rot90(arr, 2)
    elif idx == 3:
        return np.rot90(arr, 3)
    elif idx == 4:
        return np.fliplr(arr)
    elif idx == 5:
        return np.flipud(arr)
    elif idx == 6:
        return np.rot90(arr, 1).T  # diagonal flip
    elif idx == 7:
        return np.rot90(arr, 3).T  # anti-diagonal flip
    else:
        raise ValueError(f"Invalid dihedral index: {idx}")


def _start_positions_from_solution(solution: np.ndarray) -> list[tuple[int, int]]:
    """Return list of coordinates that lie on the solution path (token 'o')."""
    return list(zip(*np.where(solution == ord("o"))))


def _goal_distances(grid: np.ndarray) -> np.ndarray:
    """Compute 4-neighbor shortest-path distances from the goal across non-wall cells."""
    from collections import deque

    h, w = grid.shape
    goal_idx = np.argwhere(grid == ord("G"))
    if goal_idx.size == 0:
        return np.full_like(grid, fill_value=np.iinfo(np.int32).max, dtype=np.int32)

    goal = tuple(goal_idx[0])
    dist = np.full((h, w), fill_value=np.iinfo(np.int32).max, dtype=np.int32)
    dist[goal] = 0

    q = deque([goal])
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != ord("#"):
                if dist[nr, nc] > dist[r, c] + 1:
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))
    return dist


def _move_start_on_path(
    maze: np.ndarray, solution: np.ndarray, positions: list[tuple[int, int]]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Create maze variants where the start 'S' is placed on provided path positions.

    Returns a list of (maze_variant, solution) tuples.
    """
    variants = []
    for r, c in positions:
        maze_variant = maze.copy()
        # Clear existing start
        maze_variant[maze_variant == ord("S")] = ord(" ")
        maze_variant[r, c] = ord("S")
        variants.append((maze_variant, solution))
    return variants


def build_start_on_path_variants(
    maze: np.ndarray,
    solution: np.ndarray,
    n_random: int,
    near_goal_steps: Optional[int] = None,
    exact_steps: Optional[int] = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate variants with the start moved along the solution path.
    Always include a middle-of-path variant plus up to n_random random positions.
    If near_goal_steps is provided, restrict candidate positions to those within that
    many steps (shortest path) to the goal.
    If exact_steps is provided, keep only positions exactly that many steps from the goal.
    """
    path_positions = _start_positions_from_solution(solution)
    if not path_positions:
        return []

    dist = _goal_distances(solution)

    if exact_steps is not None:
        path_positions = [p for p in path_positions if dist[p] == exact_steps]
        if not path_positions:
            return []

    if near_goal_steps is not None:
        path_positions = [p for p in path_positions if 0 < dist[p] <= near_goal_steps]
        # If filtering removed everything, fall back to the closest path cell (excluding goal)
        if not path_positions:
            finite = [(p, dist[p]) for p in _start_positions_from_solution(solution) if dist[p] < np.iinfo(np.int32).max and dist[p] > 0]
            if finite:
                finite.sort(key=lambda x: x[1])
                path_positions = [finite[0][0]]

    indices = set()
    # Middle-of-path
    indices.add(len(path_positions) // 2)

    if n_random > 0:
        n_samples = min(n_random, len(path_positions))
        sampled = np.random.choice(len(path_positions), size=n_samples, replace=False)
        indices.update(int(i) for i in sampled)

    selected_positions = [path_positions[i] for i in sorted(indices)]
    return _move_start_on_path(maze, solution, selected_positions)


def convert_subset(set_name: str, source_repo: str, output_dir: str,
                   subsample_size: Optional[int], num_aug: int,
                   preloaded_data: Optional[Tuple] = None,
                   start_on_path_copies: int = 0,
                   near_goal_steps: Optional[int] = None,
                   only_start_on_path: bool = False,
                   start_on_path_exact_steps: Optional[int] = None) -> Tuple[int, int, Optional[Tuple]]:
    """Convert a subset and return (num_groups, num_puzzles, remaining_data)."""
    
    # Read CSV or use preloaded data
    if preloaded_data is not None:
        inputs, labels = preloaded_data
        grid_size = inputs[0].shape[0]
    else:
        inputs = []
        labels = []
        grid_size = None
        
        # Determine source file (val uses test.csv)
        source_file = "test" if set_name == "val" else set_name
        
        with open(hf_hub_download(source_repo, f"{source_file}.csv", repo_type="dataset"), newline="") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for source, q, a, rating in reader:
                if grid_size is None:
                    n = int(len(q) ** 0.5)
                    grid_size = n
                    
                inputs.append(np.frombuffer(q.encode(), dtype=np.uint8).reshape(grid_size, grid_size))
                labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(grid_size, grid_size))

    # If subsample_size is specified, randomly sample the desired number of examples.
    remaining_data = None
    if subsample_size is not None:
        print(f"Subsampling {subsample_size} examples from {set_name} set of size {len(inputs)}")
        total_samples = len(inputs)
        if subsample_size < total_samples:
            indices = np.random.choice(total_samples, size=subsample_size, replace=False)
            mask = np.ones(total_samples, dtype=bool)
            mask[indices] = False
            remaining_indices = np.where(mask)[0]
            
            # Keep remaining data for potential next split
            remaining_data = ([inputs[i] for i in remaining_indices], [labels[i] for i in remaining_indices])
            
            inputs = [inputs[i] for i in indices]
            labels = [labels[i] for i in indices]

    # Generate dataset with augmentation
    num_augments = num_aug if set_name == "train" else 0

    all_inputs = []
    all_labels = []
    puzzle_identifiers = []
    puzzle_indices = [0]  # Start at 0
    group_indices = [0]   # Start at 0
    puzzle_id = 0
    
    for orig_inp, orig_out in zip(tqdm(inputs, desc=f"Processing {set_name}"), labels):
        for aug_idx in range(1 + num_augments):
            # First index is not augmented, rest use dihedral transforms
            if aug_idx == 0:
                inp, out = orig_inp, orig_out
            else:
                # Use dihedral transforms 1-7 for augmentation
                transform_idx = (aug_idx - 1) % 7 + 1
                inp = dihedral_transform(orig_inp, transform_idx)
                out = dihedral_transform(orig_out, transform_idx)

            # Build variants: optionally keep original, plus start-on-path variants
            variants = []
            if not only_start_on_path:
                variants.append((inp, out))
            if start_on_path_copies > 0 or only_start_on_path:
                variants.extend(
                    build_start_on_path_variants(
                        inp,
                        out,
                        start_on_path_copies,
                        near_goal_steps,
                        exact_steps=start_on_path_exact_steps,
                    )
                )

            if not variants:
                # Skip puzzles that produce no variants under the constraints
                continue

            for vinp, vout in variants:
                all_inputs.append(vinp)
                all_labels.append(vout)
                puzzle_identifiers.append(0)

                puzzle_id += 1
                puzzle_indices.append(puzzle_id)
        
        # Close the group after all augmentations of this puzzle
        group_indices.append(puzzle_id)

    num_groups = len(group_indices) - 1
    num_puzzles = len(all_inputs)

    # Build char to id mapping
    char2id = np.zeros(256, dtype=np.uint8)
    for i, c in enumerate(CHARSET):
        char2id[ord(c)] = i + 1  # 0 is PAD
    
    # To Numpy - apply char mapping
    def _seq_to_numpy(seq):
        arr = np.vstack([char2id[s.reshape(-1)] for s in seq])
        return arr
    
    inputs_arr = _seq_to_numpy(all_inputs)
    labels_arr = _seq_to_numpy(all_labels)
    puzzle_identifiers = np.array(puzzle_identifiers, dtype=np.int32)

    seq_len = grid_size * grid_size
    vocab_size = len(CHARSET) + 1  # PAD + charset

    # Metadata matching Sudoku format
    metadata = {
        "num_puzzles": num_puzzles,           # Total including augmentations
        "num_groups": num_groups,              # Unique base puzzles
        "num_augmentations": num_aug if set_name == "train" else 0,
        "mean_puzzles_per_group": num_puzzles / num_groups if num_groups > 0 else 1,
        "start_on_path_copies": start_on_path_copies,
        "near_goal_steps": near_goal_steps,
        "start_on_path_exact_steps": start_on_path_exact_steps,
        "grid_size": grid_size,
        "max_grid_size": grid_size,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
    }

    # Save
    save_dir = os.path.join(output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    np.save(os.path.join(save_dir, "all__inputs.npy"), inputs_arr)
    np.save(os.path.join(save_dir, "all__labels.npy"), labels_arr)
    np.save(os.path.join(save_dir, "all__puzzle_identifiers.npy"), puzzle_identifiers)
    np.save(os.path.join(save_dir, "all__puzzle_indices.npy"), 
            np.array(puzzle_indices, dtype=np.int32))
    np.save(os.path.join(save_dir, "all__group_indices.npy"), 
            np.array(group_indices, dtype=np.int32))
        
    print(f"✓ Saved {set_name} split:")
    print(f"  - {num_groups} groups × {1 + (num_aug if set_name == 'train' else 0)} = {num_puzzles} puzzles")
    print(f"  - inputs: {inputs_arr.shape}")
    print(f"  - labels: {labels_arr.shape}")
    
    return num_groups, num_puzzles, remaining_data


@click.command()
@click.option("--source-repo", default="sapientinc/maze-30x30-hard-1k", help="Source HuggingFace repository")
@click.option("--output-dir", default="data/maze-30x30-hard-1k", help="Output directory")
@click.option("--subsample-size", type=int, default=None, help="Subsample size for training set")
@click.option("--num-aug", type=int, default=7, help="Number of augmentations per puzzle (max 7 for dihedral)")
@click.option("--eval-ratio", type=float, default=None, help="Ratio of test.csv to use for val (remainder goes to test)")
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option("--start-on-path-copies", type=int, default=0,
              help="For each puzzle, add this many random start-on-path variants plus one middle-path variant")
@click.option("--near-goal-steps", type=int, default=None,
              help="If set, only place start positions within this many steps of the goal (shortest path over open cells)")
@click.option("--only-start-on-path", is_flag=True, default=False,
              help="If set, drop the original maze and keep only start-on-path variants")
@click.option("--start-exact-steps", type=int, default=None,
              help="If set, keep only start positions exactly this many steps from the goal; puzzles without such positions are skipped")
def preprocess_data(source_repo: str, output_dir: str, subsample_size: Optional[int],
                    num_aug: int, eval_ratio: Optional[float], seed: int,
                    start_on_path_copies: int, near_goal_steps: Optional[int],
                    only_start_on_path: bool, start_exact_steps: Optional[int]):
    np.random.seed(seed)
    
    # Dihedral group has 8 elements (indices 0-7), so max meaningful num_aug is 7
    if num_aug > 7:
        print(f"Warning: num_aug={num_aug} clamped to 7 (dihedral group has 8 unique transforms)")
        num_aug = 7
    
    num_train_groups, num_train, _ = convert_subset("train", source_repo, output_dir, 
                                                     subsample_size, num_aug,
                                                     start_on_path_copies=start_on_path_copies,
                                                     near_goal_steps=near_goal_steps,
                                                     only_start_on_path=only_start_on_path,
                                                     start_on_path_exact_steps=start_exact_steps)
    
    # Val and test sets are taken from test.csv (no leakage with training)
    eval_subsample_size = None
    if eval_ratio is not None and eval_ratio < 1.0:
        with open(hf_hub_download(source_repo, "test.csv", repo_type="dataset"), newline="") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            original_test_size = sum(1 for _ in reader)
        eval_subsample_size = int(original_test_size * eval_ratio)
        print(f"Original test.csv has {original_test_size} samples, using {eval_subsample_size} for val, remainder for test")
    
    # Generate val set, keeping remaining data for test
    num_val_groups, num_val, remaining_data = convert_subset("val", source_repo, output_dir, 
                                                              eval_subsample_size, num_aug=0,
                                                              start_on_path_copies=start_on_path_copies,
                                                              near_goal_steps=near_goal_steps,
                                                              only_start_on_path=only_start_on_path,
                                                              start_on_path_exact_steps=start_exact_steps)
    
    # Generate test set from remaining pool (skip if eval_ratio=1.0 or no remaining data)
    if remaining_data is not None and len(remaining_data[0]) > 0:
        num_test_groups, num_test, _ = convert_subset("test", source_repo, output_dir, 
                                                       None, num_aug=0,
                                                       preloaded_data=remaining_data,
                                                       start_on_path_copies=start_on_path_copies,
                                                       near_goal_steps=near_goal_steps,
                                                       only_start_on_path=only_start_on_path,
                                                       start_on_path_exact_steps=start_exact_steps)
    else:
        num_test_groups, num_test = 0, 0
        print("✓ Skipping test split (all eval data used for val)")
    
    # Infer grid size from first training file
    train_meta_path = os.path.join(output_dir, "train", "dataset.json")
    with open(train_meta_path) as f:
        train_meta = json.load(f)
    grid_size = train_meta["grid_size"]
    
    # Save global metadata
    overall_meta = {
        "mode": "maze",
        "max_grid_size": grid_size,
        "vocab_size": len(CHARSET) + 1,
        "seq_len": grid_size * grid_size,
        # Puzzle counts (including augmentations)
        "num_train": num_train,
        "num_val": num_val,
        "num_test": num_test,
        # Group counts (unique base puzzles) - USE THIS FOR STEPS/EPOCH
        "num_train_groups": num_train_groups,
        "num_val_groups": num_val_groups,
        "num_test_groups": num_test_groups,
        # Augmentation info
        "num_augmentations": num_aug,
        "seed": seed,
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(overall_meta, f, indent=2)
    
    print(f"\n✓ Dataset saved to {output_dir}")
    print(f"  Train: {num_train_groups} groups × {1 + num_aug} = {num_train} puzzles")
    print(f"  Val:   {num_val_groups} groups = {num_val} puzzles")
    print(f"  Test:  {num_test_groups} groups = {num_test} puzzles")


if __name__ == "__main__":
    preprocess_data()
