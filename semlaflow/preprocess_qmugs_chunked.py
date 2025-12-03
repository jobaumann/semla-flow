"""Memory-efficient preprocessing script for QMugs dataset (chunked processing)

This version processes molecules in smaller chunks to avoid running out of RAM.

Usage:
    # Process with 500 molecules per chunk (adjust based on your RAM)
    python -m semlaflow.preprocess_qmugs_chunked --data_path ../datasets/qmugs --chunk_size 500

    # Process with smaller chunks if still running out of memory
    python -m semlaflow.preprocess_qmugs_chunked --data_path ../datasets/qmugs --chunk_size 200

    # Use first conformer only to save even more memory
    python -m semlaflow.preprocess_qmugs_chunked --data_path ../datasets/qmugs --chunk_size 500 --max_conformers 1
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from tqdm import tqdm

from semlaflow.util.molrepr import GeometricMol, GeometricMolBatch

# Default paths
DEFAULT_STRUCTURES_DIR = "structures"
DEFAULT_SUMMARY_FILE = "summary.csv"
DEFAULT_SAVE_DIR = "smol"

# File names
SAVE_TRAIN_FILE = "train.smol"
SAVE_VAL_FILE = "val.smol"
SAVE_TEST_FILE = "test.smol"
SAVE_SPLITS_FILE = "splits.pkl"

# Default chunk size (molecules per chunk)
DEFAULT_CHUNK_SIZE = 500


def create_splits(summary_df, train_frac=0.8, val_frac=0.1, random_seed=42):
    """Create train/val/test splits from QMugs summary"""
    unique_mols = summary_df['chembl_id'].unique()
    n_total = int(len(unique_mols) * 1.0)

    n_train = int(train_frac * n_total)
    n_val = int(val_frac * n_total)

    print(f"Total unique molecules: {n_total}")
    print(f"Split sizes: train={n_train}, val={n_val}, test={n_total - n_train - n_val}")

    np.random.seed(random_seed)
    shuffled = np.random.permutation(unique_mols)

    train_ids = shuffled[:n_train].tolist()
    val_ids = shuffled[n_train:n_train + n_val].tolist()
    test_ids = shuffled[n_train + n_val:n_total].tolist()

    return train_ids, val_ids, test_ids


def load_molecules_from_sdf(chembl_ids, structures_dir, max_conformers=None, skip_errors=True, max_atoms=None):
    """Load RDKit molecules from QMugs SDF files"""
    mols = []
    errors = 0
    missing_dirs = 0
    filtered_large = 0

    for chembl_id in chembl_ids:
        mol_dir = structures_dir / chembl_id

        if not mol_dir.exists():
            missing_dirs += 1
            if not skip_errors:
                raise FileNotFoundError(f"Directory not found: {mol_dir}")
            continue

        conf_files = sorted(mol_dir.glob("conf_*.sdf"))

        if len(conf_files) == 0:
            if not skip_errors:
                raise FileNotFoundError(f"No conformer files found in {mol_dir}")
            continue

        if max_conformers is not None:
            conf_files = conf_files[:max_conformers]

        for sdf_file in conf_files:
            try:
                mol = Chem.MolFromMolFile(
                    str(sdf_file),
                    removeHs=False,
                    sanitize=True
                )

                if mol is None:
                    errors += 1
                    continue

                if mol.GetNumConformers() == 0 or not mol.GetConformer().Is3D():
                    errors += 1
                    if not skip_errors:
                        raise ValueError(f"No 3D conformer in {sdf_file}")
                    continue

                if max_atoms is not None and mol.GetNumAtoms() > max_atoms:
                    filtered_large += 1
                    continue

                mols.append(mol)

            except Exception as e:
                errors += 1
                if not skip_errors:
                    raise e
                continue

    return mols, {'errors': errors, 'missing_dirs': missing_dirs, 'filtered_large': filtered_large}


def rdkit_to_smol_batch(rdkit_mols, extract_qm_properties=False, bond_order_type="GFN2:WIBERG_BOND_ORDER"):
    """Convert list of RDKit molecules to GeometricMolBatch"""
    smol_mols = []
    errors = 0

    for mol in rdkit_mols:
        try:
            smol_mol = GeometricMol.from_rdkit(
                mol,
                extract_qm_properties=extract_qm_properties,
                bond_order_type=bond_order_type
            )
            smol_mols.append(smol_mol)
        except Exception as e:
            errors += 1
            continue

    if errors > 0:
        print(f"    {errors} molecules failed conversion")

    batch = GeometricMolBatch.from_list(smol_mols)
    return batch


def process_split_chunked(split_name, chembl_ids, structures_dir, max_conformers, max_atoms,
                          extract_qm, bond_order_type, chunk_size):
    """Process one data split in chunks to save memory

    Returns:
        GeometricMolBatch object
    """
    print(f"\nProcessing {split_name} split (chunked)...")
    print(f"  Total molecules: {len(chembl_ids)}")
    print(f"  Chunk size: {chunk_size} molecules")

    # Process in chunks
    all_batches = []
    total_conformers = 0
    total_errors = 0
    total_missing = 0
    total_filtered = 0

    n_chunks = (len(chembl_ids) + chunk_size - 1) // chunk_size

    for i in tqdm(range(n_chunks), desc=f"Processing {split_name} chunks"):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(chembl_ids))
        chunk_ids = chembl_ids[start_idx:end_idx]

        # Load this chunk
        rdkit_mols, stats = load_molecules_from_sdf(
            chunk_ids,
            structures_dir,
            max_conformers=max_conformers,
            max_atoms=max_atoms,
            skip_errors=True
        )

        total_errors += stats['errors']
        total_missing += stats['missing_dirs']
        total_filtered += stats['filtered_large']

        if len(rdkit_mols) == 0:
            continue

        # Convert to smol
        smol_batch = rdkit_to_smol_batch(
            rdkit_mols,
            extract_qm_properties=extract_qm,
            bond_order_type=bond_order_type
        )

        all_batches.append(smol_batch)
        total_conformers += len(smol_batch)

        # Clear memory
        del rdkit_mols
        del smol_batch

    # Report stats
    if total_missing > 0:
        print(f"  Warning: {total_missing} molecule directories not found")
    if total_errors > 0:
        print(f"  Warning: {total_errors} conformers failed to load or validate")
    if total_filtered > 0:
        print(f"  Filtered: {total_filtered} conformers with >{max_atoms} atoms")

    print(f"  Combining {len(all_batches)} chunks...")

    # Combine all chunks into one batch
    if len(all_batches) == 0:
        raise ValueError(f"No molecules successfully processed for {split_name} split!")

    # Efficiently combine batches using collate method
    combined_batch = GeometricMolBatch.collate(all_batches)

    print(f"  {split_name.capitalize()} complete: {len(combined_batch)} conformers")

    return combined_batch


def main(args):
    # Setup paths
    data_path = Path(args.data_path)
    structures_dir = data_path / args.structures_dir
    summary_file = data_path / args.summary_file
    save_dir = data_path / args.save_dir

    # Validate input paths
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")
    if not structures_dir.exists():
        raise FileNotFoundError(f"Structures directory not found: {structures_dir}")
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")

    # Create output directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # Disable RDKit warnings
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    print("=" * 60)
    print("QMugs Preprocessing to Smol Format (Chunked)")
    print("=" * 60)
    print(f"Data path: {data_path}")
    print(f"Structures: {structures_dir}")
    print(f"Summary: {summary_file}")
    print(f"Output: {save_dir}")
    print(f"Chunk size: {args.chunk_size} molecules")
    print(f"Max conformers per molecule: {args.max_conformers if args.max_conformers else 'all'}")
    print(f"Max atoms per molecule: {args.max_atoms if args.max_atoms else 'unlimited'}")
    print(f"Extract QM properties: {'Yes' if args.extract_qm else 'No'}")
    if args.extract_qm:
        print(f"Bond order type: {args.bond_order_type}")

    # Load summary and create splits
    print("\nLoading summary.csv...")
    summary_df = pd.read_csv(summary_file)
    print(f"Summary contains {len(summary_df)} total conformers")

    print("\nCreating train/val/test splits...")
    train_ids, val_ids, test_ids = create_splits(
        summary_df,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        random_seed=args.random_seed
    )

    # Limit molecules per split if requested (for testing)
    if args.max_mols_per_split is not None:
        print(f"\n⚠️  Limiting to {args.max_mols_per_split} molecules per split (test mode)")
        train_ids = train_ids[:args.max_mols_per_split]
        val_ids = val_ids[:args.max_mols_per_split]
        test_ids = test_ids[:args.max_mols_per_split]

    # Save splits for reproducibility
    splits_path = save_dir / SAVE_SPLITS_FILE
    with open(splits_path, 'wb') as f:
        pickle.dump({
            'train_ids': train_ids,
            'val_ids': val_ids,
            'test_ids': test_ids,
            'args': vars(args)
        }, f)
    print(f"\nSaved splits to {splits_path}")

    # Process each split (chunked)
    train_batch = process_split_chunked(
        "train", train_ids, structures_dir, args.max_conformers,
        args.max_atoms, args.extract_qm, args.bond_order_type, args.chunk_size
    )

    val_batch = process_split_chunked(
        "val", val_ids, structures_dir, args.max_conformers,
        args.max_atoms, args.extract_qm, args.bond_order_type, args.chunk_size
    )

    test_batch = process_split_chunked(
        "test", test_ids, structures_dir, args.max_conformers,
        args.max_atoms, args.extract_qm, args.bond_order_type, args.chunk_size
    )

    # Save to disk
    print("\n" + "=" * 60)
    print("Saving smol files...")
    print("=" * 60)

    train_path = save_dir / SAVE_TRAIN_FILE
    val_path = save_dir / SAVE_VAL_FILE
    test_path = save_dir / SAVE_TEST_FILE

    print(f"Writing {train_path.name}... ", end='', flush=True)
    train_bytes = train_batch.to_bytes()
    train_path.write_bytes(train_bytes)
    print(f"✓ ({len(train_bytes) / (1024**2):.1f} MB)")

    print(f"Writing {val_path.name}... ", end='', flush=True)
    val_bytes = val_batch.to_bytes()
    val_path.write_bytes(val_bytes)
    print(f"✓ ({len(val_bytes) / (1024**2):.1f} MB)")

    print(f"Writing {test_path.name}... ", end='', flush=True)
    test_bytes = test_batch.to_bytes()
    test_path.write_bytes(test_bytes)
    print(f"✓ ({len(test_bytes) / (1024**2):.1f} MB)")

    # Summary
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"Train: {len(train_batch)} conformers ({len(train_ids)} molecules)")
    print(f"Val:   {len(val_batch)} conformers ({len(val_ids)} molecules)")
    print(f"Test:  {len(test_batch)} conformers ({len(test_ids)} molecules)")
    print(f"\nOutput directory: {save_dir}/")
    print("\nReady for training with:")
    print(f"  python -m semlaflow.train --data_path {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess QMugs dataset to smol format (memory-efficient chunked version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to QMugs dataset directory")

    # Chunking option
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"Number of molecules to process per chunk (default: {DEFAULT_CHUNK_SIZE})")

    # Optional paths
    parser.add_argument("--structures_dir", type=str, default=DEFAULT_STRUCTURES_DIR,
                        help="Name of structures subdirectory")
    parser.add_argument("--summary_file", type=str, default=DEFAULT_SUMMARY_FILE,
                        help="Name of summary CSV file")
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR,
                        help="Output directory name for smol files")

    # Dataset options
    parser.add_argument("--max_conformers", type=int, default=None,
                        help="Maximum conformers per molecule (default: all)")
    parser.add_argument("--max_atoms", type=int, default=None,
                        help="Maximum atoms per molecule (default: None)")
    parser.add_argument("--max_mols_per_split", type=int, default=None,
                        help="Limit molecules per split (for testing, default: no limit)")

    # QM properties extraction
    parser.add_argument("--extract_qm", action="store_true",
                        help="Extract QM properties (bond orders, energies, etc.) from SDF files")
    parser.add_argument("--bond_order_type", type=str, default="GFN2:WIBERG_BOND_ORDER",
                        choices=["GFN2:WIBERG_BOND_ORDER", "DFT:MAYER_BOND_ORDER", "DFT:WIBERG_LOWDIN_BOND_ORDER"],
                        help="Which bond order type to extract (default: GFN2:WIBERG_BOND_ORDER)")

    # Split options
    parser.add_argument("--train_frac", type=float, default=0.8,
                        help="Fraction of data for training (default: 0.8)")
    parser.add_argument("--val_frac", type=float, default=0.1,
                        help="Fraction of data for validation (default: 0.1)")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for splits (default: 42)")

    # Other options
    parser.add_argument("--verbose", action="store_true",
                        help="Show RDKit warnings")

    args = parser.parse_args()
    main(args)
