scene_name=$1;
modelname="${scene_name}_slow";

iterations=(5000 10000 15000 20000 30000 45000 70000);
iterations=(45000 70000);

for iteration in ${iterations[@]};
    do
    python train.py -s "../data/${scene_name}/" --model_path "output/${modelname}_textured_${iteration}" --init_gaussian_ply "output/${modelname}/point_cloud/iteration_${iteration}/point_cloud.ply" --textured --start_gaussians 5000 --densification_interval 100000 --densify_grad_threshold 0.0001 --max_gaussians 500000 --lambda_depth=0.0 --initialisation=depth --iterations 1000 --densify_until_iter 100000 --loss_version $2 --num_train_images $3;

    python render.py --model_path "output/${modelname}_textured_${iteration}" --iteration 1 --mode texture --blend_mode="scores2" --eval --skip_train
    python render.py --model_path "output/${modelname}_textured_${iteration}" --iteration 300 --mode texture --blend_mode="scores2" --eval --skip_train
    python render.py --model_path "output/${modelname}_textured_${iteration}" --iteration 500 --mode texture --blend_mode="scores2" --eval --skip_train
    python render.py --model_path "output/${modelname}_textured_${iteration}" --iteration 1000 --mode texture --blend_mode="scores2" --eval --skip_train

    python metrics.py --model_paths "output/${modelname}_textured_${iteration}"
done;
