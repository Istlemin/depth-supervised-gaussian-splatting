cmd="python train.py -s ../data/redwood_proc/00002/ --eval --num_train_images 3 --densification_interval 100 --densify_grad_threshold 0.0002 --iterations=8000"

$cmd --model_path output/table_images3_depth0.3_rand --initialisation=random --lambda_depth=0.3
$cmd --model_path output/table_images3_depth0.3_rand --initialisation=depth --lambda_depth=0.3
$cmd --model_path output/table_images3_depth0.3_minvis0 --min_visibility=0 --lambda_depth=0.3
$cmd --model_path output/table_images3_depth0.3_minvis1 --min_visibility=1 --lambda_depth=0.3
$cmd --model_path output/table_images3_depth0.3_minvis2 --min_visibility=2 --lambda_depth=0.3
$cmd --model_path output/table_images3_depth0.3_minvis2 --min_visibility=2 --lambda_depth=0.3