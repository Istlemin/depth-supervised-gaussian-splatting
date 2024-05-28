scene_name=$1;
modelname="${scene_name}_slow";

#loss_versions=(0 1 2 3 4 5 6);
#loss_versions=(21 24 26);
loss_versions=(4);

for loss_version in ${loss_versions[@]};
    do
    python train.py -s "../data/${scene_name}/" --model_path "output/textured_train_test/${modelname}_textured_${loss_version}" --init_gaussian_ply "output/${modelname}/point_cloud/iteration_10000/point_cloud.ply" --textured --start_gaussians 5000 --densification_interval 100000 --densify_grad_threshold 0.0001 --max_gaussians 500000 --lambda_depth=0.0 --initialisation=depth --iterations 1500 --densify_until_iter 100000 --loss_version $loss_version --num_train_images $2;

    python render.py --model_path "output/textured_train_test/${modelname}_textured_${loss_version}" --iteration 1 --mode texture --blend_mode="scores2" --eval
    python render.py --model_path "output/textured_train_test/${modelname}_textured_${loss_version}" --iteration 100 --mode texture --blend_mode="scores2" --eval
    python render.py --model_path "output/textured_train_test/${modelname}_textured_${loss_version}" --iteration 200 --mode texture --blend_mode="scores2" --eval
    python render.py --model_path "output/textured_train_test/${modelname}_textured_${loss_version}" --iteration 300 --mode texture --blend_mode="scores2" --eval
    python render.py --model_path "output/textured_train_test/${modelname}_textured_${loss_version}" --iteration 400 --mode texture --blend_mode="scores2" --eval
    python render.py --model_path "output/textured_train_test/${modelname}_textured_${loss_version}" --iteration 500 --mode texture --blend_mode="scores2" --eval
    python render.py --model_path "output/textured_train_test/${modelname}_textured_${loss_version}" --iteration 700 --mode texture --blend_mode="scores2" --eval
    python render.py --model_path "output/textured_train_test/${modelname}_textured_${loss_version}" --iteration 900 --mode texture --blend_mode="scores2" --eval
    python render.py --model_path "output/textured_train_test/${modelname}_textured_${loss_version}" --iteration 1000 --mode texture --blend_mode="scores2" --eval
    python render.py --model_path "output/textured_train_test/${modelname}_textured_${loss_version}" --iteration 1100 --mode texture --blend_mode="scores2" --eval
    python render.py --model_path "output/textured_train_test/${modelname}_textured_${loss_version}" --iteration 1300 --mode texture --blend_mode="scores2" --eval
    python render.py --model_path "output/textured_train_test/${modelname}_textured_${loss_version}" --iteration 1500 --mode texture --blend_mode="scores2" --eval

    # python render.py --model_path "output/textured_train_test/${modelname}_textured_${loss_version}" --iteration 1 --mode texture --blend_mode="scores2" --eval
    # python render.py --model_path "output/textured_train_test/${modelname}_textured_${loss_version}" --iteration 500 --mode texture --blend_mode="scores2" --eval
    # python render.py --model_path "output/textured_train_test/${modelname}_textured_${loss_version}" --iteration 1000 --mode texture --blend_mode="scores2" --eval
    # python render.py --model_path "output/textured_train_test/${modelname}_textured_${loss_version}" --iteration 1500 --mode texture --blend_mode="scores2" --eval
    
    # python metrics.py --model_paths "output/textured_train_test/${modelname}_textured_${loss_version}"

    # python train.py -s "../data/${scene_name}/" --model_path "output/textured_train_test/${modelname}_textured_seed1_${loss_version}" --init_gaussian_ply "output/${modelname}/point_cloud/iteration_10000/point_cloud.ply" --textured --start_gaussians 5000 --densification_interval 100000 --densify_grad_threshold 0.0001 --max_gaussians 500000 --lambda_depth=0.0 --initialisation=depth --iterations 1500 --densify_until_iter 100000 --loss_version $loss_version --seed 1;

    # python render.py --model_path "output/textured_train_test/${modelname}_textured_seed1_${loss_version}" --iteration 1 --mode texture --blend_mode="scores2" --eval
    # python render.py --model_path "output/textured_train_test/${modelname}_textured_seed1_${loss_version}" --iteration 500 --mode texture --blend_mode="scores2" --eval
    # python render.py --model_path "output/textured_train_test/${modelname}_textured_seed1_${loss_version}" --iteration 1000 --mode texture --blend_mode="scores2" --eval
    # python render.py --model_path "output/textured_train_test/${modelname}_textured_seed1_${loss_version}" --iteration 1500 --mode texture --blend_mode="scores2" --eval
    
    # python metrics.py --model_paths "output/textured_train_test/${modelname}_textured_seed1_${loss_version}"
done;
