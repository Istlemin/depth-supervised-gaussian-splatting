modelname=$1;

echo "model: ${modelname}";

python train.py -s $scene --model_path output/${modelname} --start_gaussians 3000 --densification_interval 1000 --densify_grad_threshold 0.0003 --max_gaussians 500000 --lambda_depth=0.0 --initialisation=depth --iterations 30000 --densify_until_iter 100000 --num_train_images $2
python train.py -s $scene --model_path output/${modelname}_depth --start_gaussians 3000 --densification_interval 2000 --densify_grad_threshold 0.0003 --max_gaussians 500000 --lambda_depth=0.8 --initialisation=depth --iterations 30000 --densify_until_iter 100000 --num_train_images $2

iterations=(1000 2000 3000 4000 5000 7000 10000 13000 16000 20000 25000 30000 37000 45000 55000 70000);

for iteration in ${iterations[@]};
    do python render.py --eval --skip_train --model_path output/${modelname}/ --iteration $iteration;
done;
python metrics.py --model_paths output/${modelname}/

for iteration in ${iterations[@]};
    do python render.py --eval --skip_train --model_path output/${modelname}_depth/ --iteration $iteration --mode texture --blend_mode="scores2";
done;
python metrics.py --model_paths output/${modelname}_depth/