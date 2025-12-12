"""Preprocessing script for QMugs dataset (SDF files) to smol format

QMugs dataset structure:
    structures/
        CHEMBL*/
            conf_00.sdf
            conf_01.sdf
            ...
    summary.csv

This script:
1. Reads summary.csv to get molecule IDs
2. Creates train/val/test splits (80/10/10)
3. Loads SDF files for each split
4. Converts to smol format
5. Saves train.smol, val.smol, test.smol

Usage:
    # Full dataset (slow! ~665k molecules)
    python -m semlaflow.preprocess_qmugs --data_path ../datasets/qmugs

    # Test with small subset (fast iteration)
    python -m semlaflow.preprocess_qmugs --data_path ../datasets/qmugs --max_mols_per_split 100 --max_conformers 1

    # Use first conformer only (like GEOM-Drugs)
    python -m semlaflow.preprocess_qmugs --data_path ../datasets/qmugs --max_conformers 1
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
SAVE_SPLITS_FILE = "splits.pkl"  # Save the splits for reproducibility


def create_splits(summary_df, train_frac=0.8, val_frac=0.1, random_seed=42):
    """Create train/val/test splits from QMugs summary

    Args:
        summary_df: DataFrame with 'chembl_id' column
        train_frac: Fraction for training (default 0.8)
        val_frac: Fraction for validation (default 0.1)
        random_seed: Random seed for reproducibility

    Returns:
        train_ids, val_ids, test_ids: Lists of chembl_ids for each split
    """
    # Get unique molecules (chembl_ids)
    unique_mols = summary_df['chembl_id'].unique()
    n_total = len(unique_mols)

    n_train = int(train_frac * n_total)
    n_val = int(val_frac * n_total)
    # test gets the remainder

    print(f"Total unique molecules: {n_total}")
    print(f"Split sizes: train={n_train}, val={n_val}, test={n_total - n_train - n_val}")

    # Shuffle and split
    np.random.seed(random_seed)
    shuffled = np.random.permutation(unique_mols)

    train_ids = shuffled[:n_train].tolist()
    val_ids = shuffled[n_train:n_train + n_val].tolist()
    test_ids = shuffled[n_train + n_val:].tolist()

    return train_ids, val_ids, test_ids


def load_molecules_from_sdf(chembl_ids, structures_dir, max_conformers=None, skip_errors=True, max_atoms=None):
    """Load RDKit molecules from QMugs SDF files

    Args:
        chembl_ids: List of ChEMBL IDs to load
        structures_dir: Path to structures directory
        max_conformers: Maximum conformers per molecule (None = all)
        skip_errors: Skip molecules that fail to load
        max_atoms: Maximum number of atoms (None = no limit)

    Returns:
        List of RDKit molecules with 3D coordinates
    """
    mols = []
    errors = 0
    missing_dirs = 0
    filtered_large = 0

    for chembl_id in tqdm(chembl_ids, desc="Loading molecules"):
        mol_dir = structures_dir / chembl_id

        if not mol_dir.exists():
            missing_dirs += 1
            if not skip_errors:
                raise FileNotFoundError(f"Directory not found: {mol_dir}")
            continue

        # Get conformer SDF files
        conf_files = sorted(mol_dir.glob("conf_*.sdf"))

        if len(conf_files) == 0:
            if not skip_errors:
                raise FileNotFoundError(f"No conformer files found in {mol_dir}")
            continue

        # Limit number of conformers if specified
        if max_conformers is not None:
            conf_files = conf_files[:max_conformers]

        # Load each conformer
        for sdf_file in conf_files:
            try:
                # Use SDMolSupplier to preserve SDF properties (bond orders, etc.)
                supplier = Chem.SDMolSupplier(
                    str(sdf_file),
                    removeHs=False,  # Keep hydrogens
                    sanitize=True
                )
                mol = supplier[0] if len(supplier) > 0 else None

                if mol is None:
                    errors += 1
                    continue

                # Check that we have 3D coordinates
                if mol.GetNumConformers() == 0 or not mol.GetConformer().Is3D():
                    errors += 1
                    if not skip_errors:
                        raise ValueError(f"No 3D conformer in {sdf_file}")
                    continue

                # Filter by size if specified
                if max_atoms is not None and mol.GetNumAtoms() > max_atoms:
                    filtered_large += 1
                    continue

                mols.append(mol)

            except Exception as e:
                errors += 1
                if not skip_errors:
                    raise e
                continue

    if missing_dirs > 0:
        print(f"  Warning: {missing_dirs} molecule directories not found")
    if errors > 0:
        print(f"  Warning: {errors} conformers failed to load or validate")
    if filtered_large > 0:
        print(f"  Filtered: {filtered_large} conformers with >{max_atoms} atoms")

    return mols


def rdkit_to_smol_batch(rdkit_mols, extract_qm_properties=False, bond_order_type="DFT:MAYER_BOND_ORDER"):
    """Convert list of RDKit molecules to GeometricMolBatch

    Args:
        rdkit_mols: List of RDKit molecules
        extract_qm_properties: Whether to extract QM properties from SDF
        bond_order_type: Which bond order type to extract

    Returns:
        GeometricMolBatch object
    """
    smol_mols = []
    errors = 0

    for mol in tqdm(rdkit_mols, desc="Converting to smol"):
        try:
            smol_mol = GeometricMol.from_rdkit(
                mol,
                extract_qm_properties=extract_qm_properties,
                bond_order_type=bond_order_type
            )
            smol_mols.append(smol_mol)
        except Exception as e:
            errors += 1
            print(f"  Warning: Failed to convert molecule: {e}")
            continue

    if errors > 0:
        print(f"  {errors} molecules failed conversion")

    batch = GeometricMolBatch.from_list(smol_mols)
    return batch


def process_split(split_name, chembl_ids, structures_dir, max_conformers, max_atoms=None,
                  extract_qm=False, bond_order_type="DFT:MAYER_BOND_ORDER", skip_errors=True):
    """Process one data split

    Args:
        split_name: Name of split ("train", "val", or "test")
        chembl_ids: List of ChEMBL IDs for this split
        structures_dir: Path to structures directory
        max_conformers: Maximum conformers per molecule
        max_atoms: Maximum number of atoms per molecule
        extract_qm: Whether to extract QM properties
        bond_order_type: Which bond order type to extract
        skip_errors: Whether to skip loading errors

    Returns:
        GeometricMolBatch object
    """
    print(f"\nProcessing {split_name} split...")
    print(f"  Loading {len(chembl_ids)} molecules from SDF files...")

    rdkit_mols = load_molecules_from_sdf(
        chembl_ids,
        structures_dir,
        max_conformers=max_conformers,
        max_atoms=max_atoms,
        skip_errors=skip_errors
    )

    print(f"  Successfully loaded {len(rdkit_mols)} conformers")
    print(f"  Converting to smol format...")

    smol_batch = rdkit_to_smol_batch(
        rdkit_mols,
        extract_qm_properties=extract_qm,
        bond_order_type=bond_order_type
    )

    print(f"  {split_name.capitalize()} batch complete: {len(smol_batch)} molecules")

    return smol_batch


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
    print("QMugs Preprocessing to Smol Format")
    print("=" * 60)
    print(f"Data path: {data_path}")
    print(f"Structures: {structures_dir}")
    print(f"Summary: {summary_file}")
    print(f"Output: {save_dir}")
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

    # Process each split
    train_batch = process_split("train", train_ids, structures_dir, args.max_conformers, args.max_atoms,
                                args.extract_qm, args.bond_order_type)
    val_batch = process_split("val", val_ids, structures_dir, args.max_conformers, args.max_atoms,
                              args.extract_qm, args.bond_order_type)
    test_batch = process_split("test", test_ids, structures_dir, args.max_conformers, args.max_atoms,
                               args.extract_qm, args.bond_order_type)

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
        description="Preprocess QMugs dataset to smol format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to QMugs dataset directory")

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
    parser.add_argument("--max_atoms", type=int, default=192,
                        help="Maximum atoms per molecule (default: 192, matches GEOM-Drugs)")
    parser.add_argument("--max_mols_per_split", type=int, default=None,
                        help="Limit molecules per split (for testing, default: no limit)")

    # QM properties extraction
    parser.add_argument("--extract_qm", action="store_true",
                        help="Extract QM properties (bond orders, energies, etc.) from SDF files")
    parser.add_argument("--bond_order_type", type=str, default="DFT:MAYER_BOND_ORDER",
                        choices=["GFN2:WIBERG_BOND_ORDER", "DFT:MAYER_BOND_ORDER", "DFT:WIBERG_LOWDIN_BOND_ORDER"],
                        help="Which bond order type to extract (default: DFT:MAYER_BOND_ORDER)")

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
