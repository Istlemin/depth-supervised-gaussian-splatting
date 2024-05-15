modelname=$1;

echo "model: ${modelname}";

num_images=(1 5 10 20 30 40 50);
for iteration in ${num_images[@]};
    do python render.py --eval --skip_train --model_path output/${modelname}_depth/ --iteration $2 --mode texture --train_images $iteration;
done;
python metrics.py --model_paths output/${modelname}_depth/