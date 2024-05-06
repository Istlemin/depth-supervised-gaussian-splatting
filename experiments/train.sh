scene="../data/chair_background2/";
modelname="chair_background2_slow";

# python train.py -s $scene --model_path output/$modelname --start_gaussians 3000 --densification_interval 1000 --densify_grad_threshold 0.0003 --max_gaussians 500000 --lambda_depth=0.0 --initialisation=depth --iterations 30000 --densify_until_iter 100000
# python train.py -s $scene --model_path output/$modelname_depth --start_gaussians 3000 --densification_interval 2000 --densify_grad_threshold 0.0003 --max_gaussians 500000 --lambda_depth=0.8 --initialisation=depth --iterations 30000 --densify_until_iter 100000

iterations=(2000 3000 4000 5000 7000 10000 13000 16000 20000 25000 30000);

# for iteration in ${iterations[@]};
#     do python render.py --eval --model_path output/$modelname/ --iteration $iteration;
# done;
for iteration in ${iterations[@]};
    do python render.py --eval --model_path output/$modelname/ --iteration $iteration --mode texture;
done;
python metrics.py --model_paths output/$modelname/


# for iteration in ${iterations[@]};
#     do python render.py --eval --model_path output/$modelname_depth/ --iteration $iteration --mode texture;
# done;
# python metrics.py --model_paths output/$modelname_depth/