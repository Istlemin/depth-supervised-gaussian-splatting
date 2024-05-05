python render.py --eval --model_path output/chair_diffuse_slow/ --iteration 2000 --mode=texture_per_gaussian
python render.py --eval --model_path output/chair_diffuse_slow/ --iteration 5000 --mode=texture_per_gaussian
python render.py --eval --model_path output/chair_diffuse_slow/ --iteration 10000 --mode=texture_per_gaussian
python render.py --eval --model_path output/chair_diffuse_slow/ --iteration 15000 --mode=texture_per_gaussian
python render.py --eval --model_path output/chair_diffuse_slow/ --iteration 20000 --mode=texture_per_gaussian
python render.py --eval --model_path output/chair_diffuse_slow/ --iteration 29000 --mode=texture_per_gaussian
python metrics.py --model_paths output/chair_diffuse_slow/

python render.py --eval --model_path output/chair_diffuse_slow_depth/ --iteration 2000 --mode=texture_per_gaussian --blend_mode alpha
python render.py --eval --model_path output/chair_diffuse_slow_depth/ --iteration 5000 --mode=texture_per_gaussian --blend_mode alpha
python render.py --eval --model_path output/chair_diffuse_slow_depth/ --iteration 10000 --mode=texture_per_gaussian --blend_mode alpha
python render.py --eval --model_path output/chair_diffuse_slow_depth/ --iteration 15000 --mode=texture_per_gaussian --blend_mode alpha
python render.py --eval --model_path output/chair_diffuse_slow_depth/ --iteration 20000 --mode=texture_per_gaussian --blend_mode alpha
python render.py --eval --model_path output/chair_diffuse_slow_depth/ --iteration 29000 --mode=texture_per_gaussian --blend_mode alpha
python metrics.py --model_paths output/chair_diffuse_slow_depth/