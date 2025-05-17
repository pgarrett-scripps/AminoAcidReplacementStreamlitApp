from dataclasses import dataclass
from typing import Optional, Set, Tuple, List, Dict, Any, FrozenSet
import pandas as pd
from intervaltree import Interval, IntervalTree
import peptacular as pt
import logging
from functools import lru_cache
import itertools

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Replacement:
    """
    Represents an amino acid replacement with associated mass difference.

    Attributes:
        mass: Mass difference between the substitute and substitutee amino acids.
        substitute: Set of amino acid sequences that replace others.
        substitutee: Set of amino acid sequences being replaced, or None for insertions.
    """

    mass: float
    substitute: Optional[FrozenSet[str]]
    substitutee: Optional[FrozenSet[str]]

    def __post_init__(self) -> None:
        """Validate that at least one of substitute or substitutee is not None."""
        if self.substitute is None and self.substitutee is None:
            raise ValueError(
                "At least one of substitute or substitutee must be provided"
            )


@lru_cache(maxsize=256)
def get_combinations(aa_sequence: str, max_length: int) -> Set[str]:
    """
    Generate all possible amino acid combinations up to specified length.
    More efficient implementation using itertools.

    Args:
        aa_sequence: String of amino acid characters to combine.
        max_length: Maximum length of combinations to generate.

    Returns:
        Set of all possible combinations.

    Raises:
        ValueError: If max_length is less than 1.
    """
    if max_length < 1:
        raise ValueError("max_length must be at least 1")

    combinations = set()
    # Use Python's built-in combinations_with_replacement for better performance
    for i in range(1, max_length + 1):
        combinations.update(
            combo for combo in pt.combinations_with_replacement(aa_sequence, i)
        )
    return combinations


def calculate_mass_error(mass: float, error_value: float, error_type: str) -> float:
    """
    Calculate mass error based on the error type.

    Args:
        mass: The mass value to calculate error for.
        error_value: Error value to use.
        error_type: Either 'ppm' or 'da' (dalton).

    Returns:
        Calculated mass error.

    Raises:
        ValueError: If error_type is not 'ppm' or 'da'.
    """
    if error_type.lower() not in ["ppm", "da"]:
        raise ValueError("error_type must be either 'ppm' or 'da'")

    if error_type.lower() == "ppm":
        mass_error = (abs(mass) * error_value) / 1e6
    else:
        mass_error = error_value

    # Ensure minimum error value for numerical stability
    return max(mass_error, 0.0005)


def build_replacement_tree(
    amino_acids: List[str],
    error_type: str = "ppm",
    error: float = 20.0,
    max_length: int = 1,
) -> Tuple[IntervalTree, IntervalTree]:
    """
    Build interval trees for amino acid replacements and insertions/deletions.
    Optimized version with better performance characteristics.

    Args:
        amino_acids: List of amino acid residues to consider.
        error_type: Type of mass error, either 'ppm' (parts per million) or 'da' (dalton).
        error: Error value to use when building interval ranges.
        max_length: Maximum length of amino acid combinations to consider.

    Returns:
        Tuple of (insertion_tree, replacement_tree) as IntervalTree objects.

    Raises:
        ValueError: If required columns are missing or input parameters are invalid.
    """
    # Validate inputs
    if error <= 0:
        raise ValueError("error must be positive")
    if max_length < 1:
        raise ValueError("max_length must be at least 1")

    replacement_tree = IntervalTree()
    insertion_tree = IntervalTree()

    if not amino_acids:
        logger.warning("Empty amino acid list provided")
        return insertion_tree, replacement_tree

    # Convert to tuple for immutability in caching
    if isinstance(amino_acids, list):
        amino_acids = tuple(amino_acids)

    aa_sequence = "".join(amino_acids)

    # Get all combinations - using cached function
    logger.info(
        f"Generating combinations for {len(amino_acids)} amino acids up to length {max_length}"
    )
    combinations = get_combinations(aa_sequence, max_length)
    logger.info(f"Generated {len(combinations)} combinations")

    # Use a set for faster lookups when checking shared elements
    combination_set = set(combinations)

    # Precompute masses for all combinations to avoid repeated calculations
    mass_cache = {comb: pt.mass(comb, ion_type="b") for comb in combinations}

    # Build insertion/deletion tree in a single pass
    for comb in combinations:
        seq_mass = mass_cache[comb]
        mass_error = calculate_mass_error(seq_mass, error, error_type)

        # Ensure non-zero mass for interval calculation
        effective_mass = max(seq_mass, 0.0001)

        # Split sequence only once to avoid repeated operations
        split_seq = frozenset(pt.split(comb))

        # Positive mass (insertion)
        insertion_tree.add(
            Interval(
                effective_mass - mass_error,
                effective_mass + mass_error,
                Replacement(mass=seq_mass, substitute=split_seq, substitutee=None),
            )
        )

        # Negative mass (deletion)
        insertion_tree.add(
            Interval(
                -effective_mass - mass_error,
                -effective_mass + mass_error,
                Replacement(mass=-seq_mass, substitute=None, substitutee=split_seq),
            )
        )

    # Build replacement tree with optimized approach
    logger.info("Building replacement tree")

    # Convert to list for indexing
    combination_list = list(combinations)
    processed_pairs = set()  # Track processed pairs to avoid duplicates

    for i, comb1 in enumerate(combination_list):
        split_comb1 = frozenset(pt.split(comb1))
        comb1_mass = mass_cache[comb1]

        # Only compare with combinations we haven't compared with yet
        for j, comb2 in enumerate(combination_list[i + 1 :], start=i + 1):
            # Generate a deterministic hash for the pair to avoid duplicates
            pair_key = hash((min(comb1, comb2), max(comb1, comb2)))
            if pair_key in processed_pairs:
                continue

            processed_pairs.add(pair_key)

            # Skip combinations with shared elements - faster check with sets
            split_comb2 = frozenset(pt.split(comb2))
            if split_comb1.intersection(split_comb2):
                continue

            comb2_mass = mass_cache[comb2]
            mass_diff = comb2_mass - comb1_mass

            mass_error = calculate_mass_error(mass_diff, error, error_type)

            # Add comb1 → comb2 replacement
            replacement_tree.add(
                Interval(
                    mass_diff - mass_error,
                    mass_diff + mass_error,
                    Replacement(
                        mass=mass_diff, substitute=split_comb2, substitutee=split_comb1
                    ),
                )
            )

            # Add comb2 → comb1 replacement (inverse)
            replacement_tree.add(
                Interval(
                    -mass_diff - mass_error,
                    -mass_diff + mass_error,
                    Replacement(
                        mass=-mass_diff, substitute=split_comb1, substitutee=split_comb2
                    ),
                )
            )

    logger.info(f"Created insertion tree with {len(insertion_tree)} intervals")
    logger.info(f"Created replacement tree with {len(replacement_tree)} intervals")
    return insertion_tree, replacement_tree


@lru_cache(maxsize=1024)
def is_valid_replacement(peptide_comps, substitutee):
    """
    Check if the substitutee can be validly replaced in the peptide.

    Args:
        peptide_comps: The components of the peptide sequence
        substitutee: The amino acids to be replaced

    Returns:
        Boolean indicating if replacement is valid
    """
    if not peptide_comps or not substitutee:
        return True

    # Convert to list for consistent handling
    peptide_list = list(peptide_comps)
    sub_list = list(substitutee)

    # Check if all substitutees are in the peptide with proper counts
    for aa in set(sub_list):
        if peptide_list.count(aa) < sub_list.count(aa):
            return False

    return True


def find_replacements(
    mass_diff: float, replacement_tree: IntervalTree, insertion_tree: IntervalTree
) -> Dict[str, List[Replacement]]:
    """
    Find possible amino acid replacements and insertions/deletions for a given mass difference.

    Args:
        mass_diff: The mass difference to search for.
        replacement_tree: IntervalTree containing replacement information.
        insertion_tree: IntervalTree containing insertion/deletion information.

    Returns:
        Dictionary with 'replacements' and 'insertions_deletions' keys mapping to lists of Replacement objects.
    """
    replacements = list(replacement_tree[mass_diff])
    insertions_deletions = list(insertion_tree[mass_diff])

    return {
        "replacements": [interval.data for interval in replacements],
        "insertions_deletions": [interval.data for interval in insertions_deletions],
    }


if __name__ == "__main__":
    # Example usage
    aa_table = pd.DataFrame(
        {
            "Residue": [
                "A",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "K",
                "L",
                "M",
                "N",
                "P",
                "Q",
                "R",
                "S",
                "T",
                "V",
                "W",
                "Y",
            ],
        }
    )
    max_length = 2
    insertion_tree, replacement_tree = build_replacement_tree(
        aa_table, "da", 0.05, max_length
    )

    print(f"Insertion Tree size: {len(insertion_tree)} intervals")
    print(f"Replacement Tree size: {len(replacement_tree)} intervals")

    # Example search
    test_mass = 128.09  # Approximately Lysine (K) → Glutamine (Q) replacement
    results = find_replacements(test_mass, replacement_tree, insertion_tree)

    print(f"\nReplacements for mass diff {test_mass}:")
    for replacement in results["replacements"]:
        print(
            f"  {replacement.substitutee} → {replacement.substitute} (Δmass: {replacement.mass:.4f})"
        )

    print(f"\nInsertions/Deletions for mass diff {test_mass}:")
    for replacement in results["insertions_deletions"]:
        if replacement.substitute:
            print(f"  Insert: {replacement.substitute} (mass: {replacement.mass:.4f})")
        else:
            print(f"  Delete: {replacement.substitutee} (mass: {replacement.mass:.4f})")
