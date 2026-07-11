

"""Structure module initialization."""

# pylint: disable=g-importing-member
from flax_model.alphafold3.structure.bioassemblies import BioassemblyData
from flax_model.alphafold3.structure.bonds import Bonds
from flax_model.alphafold3.structure.chemical_components import ChemCompEntry
from flax_model.alphafold3.structure.chemical_components import ChemicalComponentsData
from flax_model.alphafold3.structure.chemical_components import get_data_for_ccd_components
from flax_model.alphafold3.structure.chemical_components import populate_missing_ccd_data
from flax_model.alphafold3.structure.mmcif import BondParsingError
from flax_model.alphafold3.structure.parsing import BondAtomId
from flax_model.alphafold3.structure.parsing import from_atom_arrays
from flax_model.alphafold3.structure.parsing import from_mmcif
from flax_model.alphafold3.structure.parsing import from_parsed_mmcif
from flax_model.alphafold3.structure.parsing import from_res_arrays
from flax_model.alphafold3.structure.parsing import from_sequences_and_bonds
from flax_model.alphafold3.structure.parsing import ModelID
from flax_model.alphafold3.structure.parsing import NoAtomsError
from flax_model.alphafold3.structure.parsing import SequenceFormat
from flax_model.alphafold3.structure.structure import ARRAY_FIELDS
from flax_model.alphafold3.structure.structure import AuthorNamingScheme
from flax_model.alphafold3.structure.structure import Bond
from flax_model.alphafold3.structure.structure import CascadeDelete
from flax_model.alphafold3.structure.structure import concat
from flax_model.alphafold3.structure.structure import enumerate_residues
from flax_model.alphafold3.structure.structure import fix_non_standard_polymer_residues
from flax_model.alphafold3.structure.structure import GLOBAL_FIELDS
from flax_model.alphafold3.structure.structure import make_empty_structure
from flax_model.alphafold3.structure.structure import MissingAtomError
from flax_model.alphafold3.structure.structure import MissingAuthorResidueIdError
from flax_model.alphafold3.structure.structure import multichain_residue_index
from flax_model.alphafold3.structure.structure import stack
from flax_model.alphafold3.structure.structure import Structure
from flax_model.alphafold3.structure.structure_tables import Atoms
from flax_model.alphafold3.structure.structure_tables import Chains
from flax_model.alphafold3.structure.structure_tables import Residues
