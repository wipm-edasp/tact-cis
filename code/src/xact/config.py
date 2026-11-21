"""Constants and label schema for TACT."""

INTERACTION_TYPES = ("hbond", "hydrophobic", "pi", "salt", "halogen")
N_TYPES = len(INTERACTION_TYPES)
WATER_TYPES = INTERACTION_TYPES + ("water",)
N_TYPES_WATER = len(WATER_TYPES)

# distance cutoffs (Angstrom)
CONTACT_RADIUS = 4.5      # any-contact, used for copy selection / masking
MASK_RADIUS = 6.0         # atoms farther than this from protein are unsupervised
POCKET_RADIUS = 12.0      # pocket atoms kept for the model (within this of ligand)
HBOND_DIST = 3.6
HYDROPHOBIC_DIST = 4.2
PI_DIST = 5.5
SALT_DIST = 5.0
HALOGEN_DIST = 4.0
WATER_DIST = 3.5

ELEMENTS = ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "other")
HALOGENS = ("F", "Cl", "Br", "I")

# protein side-chain chemistry used ONLY to build labels and the residue-aware
# ablation; never used by the residue-blind encoder.
CHARGED_POSITIVE = {"ARG", "LYS", "HIS"}
CHARGED_NEGATIVE = {"ASP", "GLU"}

# residue ring atoms for pi-stacking (labels only)
AROMATIC_RING_ATOMS = {
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "HIS": {"CG", "ND1", "CD2", "CE1", "NE2"},
    "HID": {"CG", "ND1", "CD2", "CE1", "NE2"},
    "HIE": {"CG", "ND1", "CD2", "CE1", "NE2"},
    "HIP": {"CG", "ND1", "CD2", "CE1", "NE2"},
}

# protein hydrogen-bond donors / acceptors by atom name
PROTEIN_DONORS = {
    "N",                                    # backbone
    "ARG:NE", "ARG:NH1", "ARG:NH2",
    "LYS:NZ", "HIS:ND1", "HIS:NE2", "HID:ND1", "HIE:NE2", "HIP:ND1", "HIP:NE2",
    "TRP:NE1",
    "SER:OG", "THR:OG1", "TYR:OH",
    "ASN:ND2", "GLN:NE2",
    "CYS:SG",
}
PROTEIN_ACCEPTORS = {
    "O", "OXT",                             # backbone
    "ASP:OD1", "ASP:OD2", "GLU:OE1", "GLU:OE2",
    "ASN:OD1", "GLN:OE1",
    "HIS:ND1", "HIS:NE2", "HID:ND1", "HIE:NE2", "HIP:ND1", "HIP:NE2",
    "SER:OG", "THR:OG1", "TYR:OH",
    "MET:SD",
}

BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}

RESIDUE_TYPES = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "OTHER",
)

SEED = 20261107
