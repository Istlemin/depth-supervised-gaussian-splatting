modelname=$1;

echo "model: ${modelname}";

# python train.py -s $scene --model_path output/${modelname} --start_gaussians 3000 --densification_interval 1000 --densify_grad_threshold 0.0003 --max_gaussians 500000 --lambda_depth=0.0 --initialisation=depth --iterations 30000 --densify_until_iter 100000
# python train.py -s $scene --model_path output/${modelname}_depth --start_gaussians 3000 --densification_interval 2000 --densify_grad_threshold 0.0003 --max_gaussians 500000 --lambda_depth=0.8 --initialisation=depth --iterations 30000 --densify_until_iter 100000

iterations=(1000 2000 3000 4000 5000 7000 10000 13000 16000 20000 25000 30000 37000 45000 55000 70000);
#iterations=(1000);
# for iteration in ${iterations[@]};
#     do python render.py --eval --skip_train --model_path output/${modelname}/ --iteration $iteration;
# done;
# for iteration in ${iterations[@]};
#     do python render.py --eval --skip_train --model_path output/${modelname}/ --iteration $iteration --mode texture;
# done;
# for iteration in ${iterations[@]};
#     do python render.py --eval --skip_train --model_path output/${modelname}/ --iteration $iteration --mode texture_per_gaussian;
# done;
# python metrics.py --model_paths output/${modelname}/



num_images=(1 5 10 20 30 40 50 70 100);
for iteration in ${num_images[@]};
    do python render.py --eval --skip_train --model_path output/${modelname}_depth/ --iteration $2 --mode texture --train_images $iteration;
done;

for iteration in ${iterations[@]};
    do python render.py --eval --skip_train --model_path output/${modelname}_depth/ --iteration $iteration --mode texture;
done;

# for iteration in ${iterations[@]};
#     do python render.py --eval --skip_train --model_path output/${modelname}_depth/ --iteration $iteration --mode texture_per_gaussian;
# done;
python metrics.py --model_paths output/${modelname}_depth/