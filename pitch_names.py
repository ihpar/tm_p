import os
from collections import Counter
from collections import defaultdict

from constants import *
from file_parser import FileParser


PC_ID_TO_NAME = {v: k for k, v in PC_VOCAB.items()}


def get_sharp_name(pc_id, acc_id):
    """
    Pitch name of a note as the model stores it: every accidental is a
    sharp, counted in kommas above the natural pitch class. No octave.

    (pc "A", acc "#5") -> "A#5"
    (pc "D", no acc)   -> "D"
    rest               -> "Rest"
    """
    if pc_id == REST_ID:
        return "Rest"

    pc_name = PC_ID_TO_NAME[pc_id]

    if acc_id <= 1:
        return pc_name

    acc_kommas = acc_id - 1
    return f"{pc_name}#{acc_kommas}"


def get_symbtr_name(pc_id, acc_id, spelling_map):
    """
    Pitch name of a note as it is most often written in the SymbTr files
    of one makam.
    "A#5" -> "Bb4"

    spelling_map: the spellings of one makam, {"A#5": "Bb4", ...}
                  (one entry of build_symbtr_spelling_map()).
                  If None, the sharp name is returned.
    """
    sharp_name = get_sharp_name(pc_id, acc_id)

    if spelling_map is None:
        return sharp_name

    return spelling_map.get(sharp_name, sharp_name)


def build_symbtr_spelling_map(data_dir=SYMBTR_TXT_DIR):
    """
    For every makam and every sharp name, finds the spelling that is most
    frequent in the SymbTr txt files of that makam (octave removed).

    The same pitch can be written differently in different makams: the
    pitch 4 kommas above G is "G#4" in the Hicaz scores and "Ab5" in the
    Kurdili Hicazkar scores.

    The makam of a file is the first part of its name:
    "hicaz--sarki--aksak--..." -> "hicaz"

    returns {"hicaz": {"A#5": "Bb4", "G#4": "G#4", ...},
             "kurdilihicazkar": {"G#4": "Ab5", ...}, ...}
    """
    if not os.path.isdir(data_dir):
        print(f"{data_dir} not found, using the sharp names")
        return {makam: None for makam in MAKAM_NAMES}

    file_parser = FileParser()

    # makam -> sharp name -> Counter of the original spellings
    spelling_counts = {}
    for makam in MAKAM_NAMES:
        spelling_counts[makam] = defaultdict(Counter)

    for f_name in os.listdir(data_dir):
        if not f_name.endswith(".txt"):
            continue

        makam = f_name.split("--")[0]
        if makam not in spelling_counts:
            continue

        f_path = os.path.join(data_dir, f_name)
        with open(f_path, encoding="utf-8", errors="replace") as in_file:
            for line in in_file:
                tokens = line.rstrip("\n").split("\t")

                if len(tokens) < 5:
                    continue
                if tokens[1] not in file_parser.note_line_codes:
                    continue

                # Some examples: D5, Es, E5b4, F5#4
                note = tokens[3]
                if note in ["", "Es"]:
                    continue

                pc, acc, _ = file_parser.parse_pitch(note)
                if acc is None:
                    sharp_name = pc
                else:
                    sharp_name = f"{pc}{acc}"

                # original spelling without the octave: "B4b1" -> "Bb1"
                original_spelling = note[0] + note[2:]
                spelling_counts[makam][sharp_name][original_spelling] += 1

    spelling_map = {}
    for makam in MAKAM_NAMES:
        spelling_map[makam] = {}
        for sharp_name, counts in spelling_counts[makam].items():
            spelling_map[makam][sharp_name] = counts.most_common(1)[0][0]

    return spelling_map


def with_accidental_symbols(pitch_name):
    """
    Replaces "b" and "#" with the flat and sharp symbols in a figure label.
    "Bb4" -> "B♭4" , "C#4" -> "C♯4"  (written in matplotlib mathtext)
    """
    if len(pitch_name) > 1 and pitch_name[1] in "b#":
        if pitch_name[1] == "b":
            symbol = "$\\flat$"
        else:
            symbol = "$\\sharp$"
        return pitch_name[0] + symbol + pitch_name[2:]

    return pitch_name
