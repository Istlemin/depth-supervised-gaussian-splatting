python render.py --eval --model_path output/chair_diffuse_slow/ --iteration 29000 --mode=texture_per_gaussian
python render.py --eval --model_path output/chair_diffuse_slow/ --iteration 29000 --mode=texture
python render.py --eval --model_path output/chair_diffuse_slow/ --iteration 29000
python render.py --eval --model_path output/chair_diffuse_slow_depth/ --iteration 29000 --mode=texture_per_gaussian
python render.py --eval --model_path output/chair_diffuse_slow_depth/ --iteration 29000 --mode=texture
python render.py --eval --model_path output/chair_diffuse_slow_depth/ --iteration 29000
python metrics.py --model_paths output/chair_diffuse_slow/
python metrics.py --model_paths output/chair_diffuse_slow_depth/