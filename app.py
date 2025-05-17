import peptacular as pt
import streamlit as st
import pandas as pd
import base64
import json

from util import build_replacement_tree, is_valid_replacement
import streamlit_permalink as stp


# Set the page title and layout
st.set_page_config(
    page_title="Amino Acid Replacement Calculator",
    page_icon="🔄",
    initial_sidebar_state="expanded",
)

st.title("Peptide Mass Shift Solver")
st.markdown(
    """
- **Input a peptide sequence** and the **observed mass shift** and **Amino Acids/Modifications**
- The app will calculate all possible amino acid replacements or insertions that could explain the mass difference
- Results are filtered to show only substitutions possible within your peptide sequence
"""
)


# Helper function for generating downloadable results
def get_download_link(data, filename, text):
    """Generate a link to download the data as a file"""
    json_str = json.dumps(data, indent=2, default=str)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:application/json;base64,{b64}" download="{filename}">💾 {text}</a>'
    return href


c1, c2 = st.columns(2)

with c1:
    # Error mode selection
    st.subheader("Input")
    error_mode = stp.radio(
        "Mass Tolerance Type",
        ["ppm", "Da"],
        horizontal=True,
        index=1,
        help="Select 'ppm' for parts per million or 'Da' for Dalton mass error tolerance",
    )
    error_value = stp.number_input(
        "Mass Tolerance",
        min_value=0.0,
        value=10.0,
        max_value=1000.0,
        help="Specify the mass error tolerance in the selected unit (ppm or Da)",
    )
    # max_length = stp.number_input("Max Length", min_value=1, max_value=100, value=1, disabled=True,
    #                           help="Maximum length of amino acid combinations to consider")
    max_length = 1

    peptide_sequence = stp.text_input(
        "Filter Sequence",
        value="PEPTIDE",
        max_chars=50,
        help="Enter the peptide sequence. Will be used to filter the results.",
    )
    mass_shift = stp.number_input(
        "Observed Mass Shift (Da)",
        min_value=-1000.0,
        max_value=1000.0,
        value=10.0,
        help="Enter the observed mass difference in Daltons that needs to be explained",
    )

with c2:

    # Input for amino acids with modifications
    st.subheader("Amino Acids")

    amino_acids: set[str] = pt.AMINO_ACIDS
    amino_acids = set(amino_acids) - {"C", "J", "I", "U", "O", "X", "B", "Z"}

    # add cystein +57.021464
    amino_acids.add("C[57.0215]")
    amino_acids.add("C[58.00548]")

    # add other popular mods
    amino_acids.add("M[15.9949]")  # Oxidation
    amino_acids.add("C[15.9949]")  # Oxidation
    amino_acids.add("H[15.9949]")  # Oxidation
    amino_acids.add("W[15.9949]")  # Oxidation

    # dehydration
    amino_acids.add("S[-18.0106]")  # Dehydration
    amino_acids.add("T[-18.0106]")  # Dehydration
    amino_acids.add("Y[-18.0106]")  # Dehydration

    amino_acids.add("S[79.9663]")  # Phosphorylation
    amino_acids.add("T[79.9663]")  # Phosphorylation
    amino_acids.add("Y[79.9663]")  # Phosphorylation
    amino_acids.add("H[79.9663]")  # Phosphorylation
    amino_acids.add("D[79.9663]")  # Phosphorylation
    amino_acids.add("C[79.9663]")  # Phosphorylation
    amino_acids.add("R[79.9663]")  # Phosphorylation

    # sulfation
    amino_acids.add("Y[79.9568]")  # Sulfation

    amino_acids.add("K[42.0106]")  # Acetylation
    amino_acids.add("C[42.0106]")  # Acetylation
    amino_acids.add("S[42.0106]")  # Acetylation

    amino_acids.add("N[0.9840]")  # Deamidation
    amino_acids.add("Q[0.9840]")  # Deamidation
    amino_acids.add("R[0.9840]")  # Deamidation

    default_aa_data = {
        "Residue": list(amino_acids),
    }

    aa_df = stp.data_editor(
        pd.DataFrame(default_aa_data), num_rows="dynamic", key="aa_editor", height=320
    )

    # limit the number of rows to 100
    if len(aa_df) > 100:
        st.warning("Too many amino acids selected. Limiting to 100.")
        aa_df = aa_df[:100]


peptide_comps = pt.split(peptide_sequence)


# Use Streamlit's cache to avoid rebuilding the tree on every interaction
@st.cache_data
def get_replacement_trees(aa_list, error_mode, error_value, max_length):
    return build_replacement_tree(aa_list, error_mode, error_value, max_length)


# Only rebuild trees when inputs affecting them change
aa_list = list(aa_df["Residue"])
insertion_tree, replacement_tree = get_replacement_trees(
    tuple(aa_list), error_mode, error_value, max_length
)

insertions = insertion_tree[mass_shift]
replacements = replacement_tree[mass_shift]


st.header("Results", divider=True)

st.subheader("Insertions/Deletions")
if insertions:

    # Create a table format for insertions
    ins_data = []
    peptide_comps_tuple = tuple(peptide_comps)

    valid_insertions = []
    # Process in batches for better performance
    for i, insertion in enumerate(insertions):

        # check if the insertion is valid (only applies to deletions, so negative mass shifts)
        if mass_shift < 0 and is_valid_replacement(
            peptide_comps_tuple, insertion.data.substitutee
        ):
            valid_insertions.append(insertion)

    for i, insertion in enumerate(valid_insertions):
        # Format the substitutee and substitute to remove frozenset representation
        sub_from = (
            ", ".join(sorted(insertion.data.substitutee))
            if isinstance(insertion.data.substitutee, (set, frozenset))
            else str(insertion.data.substitutee)
        )
        sub_to = (
            ", ".join(sorted(insertion.data.substitute))
            if isinstance(insertion.data.substitute, (set, frozenset))
            else str(insertion.data.substitute)
        )

        ins_data.append(
            {
                "From": sub_from,
                "To": sub_to,
                "Mass": f"{round(insertion.data.mass, 5)} Da",
                "Mass Δ": f"{round(mass_shift - insertion.data.mass, 5)} Da",
            }
        )

    # Sort by mass for better visualization
    valid_insertions.sort(key=lambda x: abs(x.data.mass))

    st.info(
        f"Found {len(valid_insertions)} valid insertions out of {len(insertions)} total possibilities"
    )

    # Display as a dataframe for better formatting
    if ins_data:
        st.dataframe(pd.DataFrame(ins_data), use_container_width=True, hide_index=True)
    else:
        st.warning("No insertions found.")
else:
    st.warning("No insertions found.")

st.subheader("Replacements")
if replacements:
    # Convert peptide_comps to tuple for caching to work
    peptide_comps_tuple = tuple(peptide_comps)

    valid_replacements = []
    for i, replacement in enumerate(replacements):

        # Use the cached validation function
        if is_valid_replacement(
            peptide_comps_tuple, tuple(replacement.data.substitutee)
        ):
            valid_replacements.append(replacement)

    # Show number of matching replacements
    st.info(
        f"Found {len(valid_replacements)} valid replacements out of {len(replacements)} total possibilities"
    )

    # Sort by mass for better visualization
    valid_replacements.sort(key=lambda x: abs(x.data.mass))

    # Prepare data for display in a table format
    repl_data = []
    for replacement in valid_replacements:
        # Format the substitutee and substitute to remove frozenset representation
        from_str = (
            ", ".join(sorted(replacement.data.substitutee))
            if isinstance(replacement.data.substitutee, (set, frozenset))
            else str(replacement.data.substitutee)
        )
        to_str = (
            ", ".join(sorted(replacement.data.substitute))
            if isinstance(replacement.data.substitute, (set, frozenset))
            else str(replacement.data.substitute)
        )

        repl_data.append(
            {
                "From": from_str,
                "To": to_str,
                "Mass": f"{round(replacement.data.mass, 5)} Da",
                "Mass Δ": f"{round(mass_shift - replacement.data.mass, 5)} Da",
            }
        )

    # Display as a dataframe for better formatting
    if repl_data:
        st.dataframe(pd.DataFrame(repl_data), use_container_width=True, hide_index=True)
    else:
        st.warning("No valid replacements found.")

else:
    st.warning("No replacements found.")
