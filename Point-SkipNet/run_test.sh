# Test Point-SkipNet

# Check if dataset is modelnetR
if [ "$dataset" = "modelnetR" ]; then
    python test_classification.py --dataset modelnetR --log_dir pn2_cls_ssg_new_dset_cat_rot_1

elif [ "$dataset" = "modelnet" ]; then
    python test_classification.py --dataset modelnet --log_dir pn2_cls_ssg_new_dset_cat_rot_1

else
    echo "Unsupported dataset: $dataset"
    exit 1
fi
