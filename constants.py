# ---------------------------------------------------------------------------
# Files and folders
# ---------------------------------------------------------------------------
VOCAB_FILE = "vocab.pkl"
DATASET_FILE = "dataset.pkl"
SYMBTR_TXT_DIR = "data/txt/"

SPLIT_DIR = "splits/"
RESULTS_DIR = "results/"
FIGS_DIR = "figs/"

# a tiny dry run of every step writes here (QUICK_TEST = True in the notebooks)
QUICK_TEST_RESULTS_DIR = "results_quick_test/"
QUICK_TEST_FIGS_DIR = "figs_quick_test/"

# ---------------------------------------------------------------------------
# Vocabularies (the same ids as in vocab.pkl; the rest token is called
# "Es" there, as in the SymbTr files)
# ---------------------------------------------------------------------------
PC_VOCAB = {
    "PAD": 0, "Rest": 1,
    "G": 2, "A": 3, "B": 4, "C": 5, "D": 6, "E": 7, "F": 8
}
# accidental "#k" = k kommas above the natural pitch class
ACC_VOCAB = {
    "PAD": 0, None: 1,
    "#1": 2, "#2": 3, "#3": 4, "#4": 5, "#5": 6, "#6": 7, "#7": 8, "#8": 9
}
OCT_VOCAB = {"PAD": 0, None: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
MEAS_VOCAB = {"PAD": 0, "start": 1, "middle": 2, "end": 3}
MAKAM_VOCAB = {
    "hicaz": 0, "nihavent": 1, "ussak": 2, "rast": 3,
    "huzzam": 4, "segah": 5, "huseyni": 6, "mahur": 7,
    "hicazkar": 8, "kurdilihicazkar": 9, "muhayyer": 10, "saba": 11
}

PAD_ID = 0
REST_ID = PC_VOCAB["Rest"]
NUM_CLASSES = len(MAKAM_VOCAB)

# makam ids in label order: MAKAM_NAMES[label] -> makam id
MAKAM_NAMES = [
    "hicaz", "nihavent", "ussak", "rast",
    "huzzam", "segah", "huseyni", "mahur",
    "hicazkar", "kurdilihicazkar", "muhayyer", "saba"
]

# makam names as they are written in the figures and tables
MAKAM_DISPLAY_NAMES = {
    "hicaz": "Hicaz",
    "nihavent": "Nihavent",
    "ussak": "Uşşak",
    "rast": "Rast",
    "huzzam": "Hüzzam",
    "segah": "Segah",
    "huseyni": "Hüseyni",
    "mahur": "Mahur",
    "hicazkar": "Hicazkar",
    "kurdilihicazkar": "Kürdili Hicazkar",
    "muhayyer": "Muhayyer",
    "saba": "Saba"
}

MEAS_ID_TO_NAME = {1: "start", 2: "middle", 3: "end"}

# ---------------------------------------------------------------------------
# Komma arithmetic (53 kommas per octave)
# ---------------------------------------------------------------------------
KOMMAS_PER_OCTAVE = 53

# komma offset of each natural pitch class from C within an octave
# (C-D 9, D-E 9, E-F 4, F-G 9, G-A 9, A-B 9, B-C 4). The octave number
# increments at C, as in the SymbTr pitch names, so C4 < G4.
# Verified against the KomaAE column of SymbTr.
PC_KOMMA_BASE = {"C": 0, "D": 9, "E": 18, "F": 22, "G": 31, "A": 40, "B": 49}

# octaves in the vocabulary (3, 4, 5, 6, 7)
LOWEST_OCTAVE = 3
HIGHEST_OCTAVE = 7
NUM_OCTAVES = 5

# Interval representation (transposition-invariant pitch feature):
# komma difference to the previous pitched note, clipped to +-INTERVAL_CLIP.
#   id 0 = PAD
#   id 1 = rest
#   id 2 = first pitched note of the piece (it has no predecessor)
#   id 3 ... 3 + 2 * INTERVAL_CLIP = interval -INTERVAL_CLIP ... +INTERVAL_CLIP
INTERVAL_CLIP = 40
INTERVAL_REST_ID = 1
INTERVAL_FIRST_NOTE_ID = 2
INTERVAL_OFFSET = 3
INTERVAL_VOCAB_SIZE = INTERVAL_OFFSET + 2 * INTERVAL_CLIP + 1

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# symbolic attributes of a note that the proposed model embeds
ALL_FEATURES = ["pc", "acc", "oct", "dur", "meas"]

# feature name -> key of the feature in a batch
FEATURE_TO_BATCH_KEY = {
    "pc": "pcs",
    "acc": "accs",
    "oct": "octs",
    "dur": "durs",
    "meas": "meas",
    "int": "ints"
}

PC_VOCAB_SIZE = 9
ACC_VOCAB_SIZE = 10
OCT_VOCAB_SIZE = 7
DUR_VOCAB_SIZE = 33
MEAS_VOCAB_SIZE = 4

EMB_DIM = 32
LSTM_HIDDEN = 128
NUM_LAYERS = 1
DROPOUT = 0.3

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
PATIENCE = 7                # early stopping
SCHEDULER_PATIENCE = 3      # ReduceLROnPlateau
SCHEDULER_FACTOR = 0.5
GRAD_CLIP_NORM = 5.0

# ---------------------------------------------------------------------------
# Evaluation protocol
# ---------------------------------------------------------------------------
# one stratified 70/15/15 split per seed
SEEDS = [0, 1, 2, 3, 4]
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

# transpositions of the test pieces, in kommas
TRANSPOSITIONS = [4, 5, 9, 13, -4, -5, -9, -13]

# random transposition range of the transposition-augmented model, in kommas
TRANSPOSE_AUG_RANGE = 13

# hyperparameter search
NUM_SEARCH_TRIALS = 20
SEARCH_SEED = 123
SEARCH_SPACE = {
    "lstm_hidden": [64, 128, 256],
    "emb_dim": [16, 32, 64],
    "learning_rate": [1e-3, 5e-4],
    "dropout": [0.2, 0.3, 0.5],
    "batch_size": [8, 16]
}

# deletion tests
DELETION_FRACTIONS = [0.05, 0.10, 0.20, 0.40]
NUM_RANDOM_DRAWS = 5

# ---------------------------------------------------------------------------
# Attention analysis
# ---------------------------------------------------------------------------
# relative duration bins: (0, 1/16], (1/16, 3/16], (3/16, 3/8], (3/8, 3/4], (3/4, 1]
DURATION_BIN_EDGES = [0.0, 0.0625, 0.1875, 0.375, 0.75, 1.01]
DURATION_BIN_LABELS = [
    "Very short (≤1/16)",
    "Short (1/16–3/16]",
    "Medium (3/16–3/8]",
    "Long (3/8–3/4]",
    "Very long (>3/4)"
]

# a note is a "high attention" note if its attention is at or above this
# percentile of the attention weights of its own piece
HIGH_ATTENTION_PERCENTILE = 95

# the test piece shown in the example figure
EXAMPLE_PIECE = "hicaz--sarki--duyek--bogazici_sen"
EXAMPLE_PIECE_SEED = 0
EXAMPLE_PIECE_NUM_PEAKS = 6
