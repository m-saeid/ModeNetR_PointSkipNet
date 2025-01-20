# Train Point-SkipNet

# Check if dataset is modelnetR
if [ "$dataset" = "modelnetR" ]; then
    python train_classification.py --model pointskipnet --dataset modelnetR --batch_size 110 --seed 1 --epoch 200 --aug_type rotation --res_type cat

elif [ "$dataset" = "modelnet" ]; then
    python train_classification.py --model pointskipnet --dataset modelnet --batch_size 110 --seed 1 --epoch 200 --aug_type rotation --res_type cat

else
    echo "Unsupported dataset: $dataset"
    exit 1
fi
