cmd="python train.py -s ../data/redwood_proc/00002/ --eval --num_train_images 3 --iterations=10000"

$cmd --model_path output/table_images3_depth0_rand --random_initialisation --lambda_depth=0
$cmd --model_path output/table_images3_depth0.5 --random_initialisation --lambda_depth=0.5
$cmd --model_path output/table_images3_depth0_minvis0 --min_visibility=0 --lambda_depth=0
$cmd --model_path output/table_images3_depth0.5 --min_visibility=0 --lambda_depth=0.5
$cmd --model_path output/table_images3_depth0_minvis1 --min_visibility=1 --lambda_depth=0
$cmd --model_path output/table_images3_depth0.5 --min_visibility=1 --lambda_depth=0.5
$cmd --model_path output/table_images3_depth0_minvis2 --min_visibility=2 --lambda_depth=0
$cmd --model_path output/table_images3_depth0.5 --min_visibility=2 --lambda_depth=0.5