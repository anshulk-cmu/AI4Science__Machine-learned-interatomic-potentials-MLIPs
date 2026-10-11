# Sourced by pilot.sh and prep_pilot_data.sh. Needs REPO and OMOL_DATA_ROOT.
# The Hydra overrides that make a run read the fixed sample lists in manifests/ instead of drawing a random sample.
LISTS=${PILOT_LISTS_DIR:-$REPO/manifests}
TRAIN_LIST=${PILOT_TRAIN_LIST:-$LISTS/pilot_train_200k.txt}
VAL_LIST=${PILOT_VAL_LIST:-$LISTS/pilot_val_5k.txt}
T=train_dataset.dataset_configs.omol.splits.train
V=val_dataset.dataset_configs.omol.splits.val
DATA_OVERRIDES=(
    "$T.sample_n=null" "$T.no_shuffle=true"
    "$T.metadata_path=$OMOL_DATA_ROOT/pilot/train_4M_index.npz"
    "$T.subset_to=[{metadata_key:index,op:in,rhv:$TRAIN_LIST}]"
    "$V.sample_n=null" "$V.no_shuffle=true"
    "$V.metadata_path=$OMOL_DATA_ROOT/pilot/val_index.npz"
    "$V.subset_to=[{metadata_key:index,op:in,rhv:$VAL_LIST}]"
)
