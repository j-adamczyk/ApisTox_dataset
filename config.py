import os

UNITS_TO_EXCLUDE = [
    "%",
    "AI %",
    "AI mg/L",
    "AI ng",
    "AI ng/g bdwt",
    "AI ppm",
    "AI ppm diet",
    "AI ug",
    "AI ug/g",
    "AI ug/g bdwt",
    "AI ug/ml",
    "ae ug/org",
    "ae ug/org/d",
    "ae ug/org:x",
    "mg",
    "mg/L",
    "mg/L diet",
    "mg/cm2",
    "mg/mg bdwt",
    "mg/ml diet",
    "ml/1000 org",
    "ng/mg bdwt",
    "ng/ul",
    "ug",
    "ug/eu",
    "ug/g",
    "ug/g bdwt",
    "ug/g diet",
    "ug/g org",
    "ug/ul",
    "ul/org",
]

RAW_DATA_DIR = "raw_data"
OUTPUTS_DIR = "outputs"

ECOTOX_FILE_PATH = os.path.join(RAW_DATA_DIR, "ecotox.csv")
PPDB_FILE_PATH = os.path.join(RAW_DATA_DIR, "ppdb.csv")
BPDB_FILE_PATH = os.path.join(RAW_DATA_DIR, "bpdb.csv")

ECOTOX_CLEANED_FILE_PATH = os.path.join(OUTPUTS_DIR, "ecotox_cleaned_data.csv")
EXCLUDED_DATA_FILE_PATH = os.path.join(OUTPUTS_DIR, "excluded_data.csv")
DATASET_FINAL_FILE_PATH = os.path.join(OUTPUTS_DIR, "dataset_final.csv")

SPLITS_DIR = os.path.join(OUTPUTS_DIR, "splits")
PLOTS_DIR = "plots"
MOSS_DATA_DIR = "moss_analysis_data"

OTHER_SOURCES_DIR = "other_sources"
BEETOX_DATA = os.path.join(OTHER_SOURCES_DIR, "beetox_raw_data.csv")
CROP_CSM_DATA = os.path.join(OTHER_SOURCES_DIR, "crop_csm_apis_raw_data.csv")
BEETOX_AI_DATA = os.path.join(OTHER_SOURCES_DIR, "beetox_ai_raw_smiles.csv")
BEETOX_AI_ANNOTATED_DATA = os.path.join(
    OTHER_SOURCES_DIR, "beetox_ai_annotated_molecules.csv"
)
