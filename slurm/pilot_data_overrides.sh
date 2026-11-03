# Sourced by pilot.sh, long_run.sh and the prep scripts. Needs REPO and OMOL_DATA_ROOT.
# The Hydra overrides that make a run read the fixed sample lists in manifests/ instead of drawing a random sample.
# PILOT_VAL_SPLIT: the split folder the validation rows come from (val for the pilot lists, train_4M for the 70/20/10 lists).
LISTS=${PILOT_LISTS_DIR:-$REPO/manifests}
TRAIN_LIST=${PILOT_TRAIN_LIST:-$LISTS/pilot_train_200k.txt}
VAL_LIST=${PILOT_VAL_LIST:-$LISTS/pilot_val_5k.txt}
VAL_SPLIT=${PILOT_VAL_SPLIT:-val}
T=train_dataset.dataset_configs.omol.splits.train
V=val_dataset.dataset_configs.omol.splits.val
DATA_OVERRIDES=(
    "$T.sample_n=null" "$T.no_shuffle=true"
    "$T.metadata_path=$OMOL_DATA_ROOT/pilot/train_4M_index.npz"
    "$T.subset_to=[{metadata_key:index,op:in,rhv:$TRAIN_LIST}]"
    "$V.src=$OMOL_DATA_ROOT/$VAL_SPLIT"
    "$V.sample_n=null" "$V.no_shuffle=true"
    "$V.metadata_path=$OMOL_DATA_ROOT/pilot/${VAL_SPLIT}_index.npz"
    "$V.subset_to=[{metadata_key:index,op:in,rhv:$VAL_LIST}]"
)
