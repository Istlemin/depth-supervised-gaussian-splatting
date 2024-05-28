
scene_names=(chair_background2 mic ship livingroom);

for scene_name in ${scene_names[@]};
    do
    python render.py --model_path output/${scene_name}_slow/ --iteration 10000 --mode depth --blend_mode="scores2" --eval --skip_train
    python render.py --model_path output/${scene_name}_slow/ --iteration 10000 --mode depth_not_normalized --blend_mode="scores2" --eval --skip_train
    python render.py --model_path output/${scene_name}_slow_depth/ --iteration 10000 --mode depth --blend_mode="scores2" --eval --skip_train
    python render.py --model_path output/${scene_name}_slow_baddepth/ --iteration 10000 --mode depth_not_normalized --blend_mode="scores2" --eval --skip_train

done;