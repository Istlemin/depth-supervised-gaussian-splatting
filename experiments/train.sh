python train.py -s ../data/chair_background/ --model_path output/chair_background_slow --start_gaussians 5000 --densification_interval 2000 --densify_grad_threshold 0.0003 --max_gaussians 500000 --lambda_depth=0.0 --initialisation=depth --iterations 70000 --densify_until_iter 100000
python train.py -s ../data/chair_background/ --model_path output/chair_background_slow_depth --start_gaussians 3000 --densification_interval 2000 --densify_grad_threshold 0.0003 --max_gaussians 500000 --lambda_depth=0.8 --initialisation=depth --iterations 70000 --densify_until_iter 100000

python render.py --eval --model_path output/chair_background_slow/ --iteration 2000 --mode texture
python render.py --eval --model_path output/chair_background_slow/ --iteration 3000 --mode texture
python render.py --eval --model_path output/chair_background_slow/ --iteration 4000 --mode texture
python render.py --eval --model_path output/chair_background_slow/ --iteration 5000 --mode texture
python render.py --eval --model_path output/chair_background_slow/ --iteration 7000 --mode texture
python render.py --eval --model_path output/chair_background_slow/ --iteration 10000 --mode texture
python render.py --eval --model_path output/chair_background_slow/ --iteration 13000 --mode texture
python render.py --eval --model_path output/chair_background_slow/ --iteration 16000 --mode texture
python render.py --eval --model_path output/chair_background_slow/ --iteration 20000 --mode texture
python render.py --eval --model_path output/chair_background_slow/ --iteration 25000 --mode texture
python render.py --eval --model_path output/chair_background_slow/ --iteration 30000 --mode texture
python render.py --eval --model_path output/chair_background_slow/ --iteration 35000 --mode texture
python render.py --eval --model_path output/chair_background_slow/ --iteration 40000 --mode texture
python render.py --eval --model_path output/chair_background_slow/ --iteration 47000 --mode texture
python render.py --eval --model_path output/chair_background_slow/ --iteration 55000 --mode texture
python render.py --eval --model_path output/chair_background_slow/ --iteration 70000 --mode texture
python metrics.py --model_paths output/chair_background_slow/

python render.py --eval --model_path output/chair_background_slow_depth/ --iteration 2000 --mode texture
python render.py --eval --model_path output/chair_background_slow_depth/ --iteration 3000 --mode texture
python render.py --eval --model_path output/chair_background_slow_depth/ --iteration 4000 --mode texture
python render.py --eval --model_path output/chair_background_slow_depth/ --iteration 5000 --mode texture
python render.py --eval --model_path output/chair_background_slow_depth/ --iteration 7000 --mode texture
python render.py --eval --model_path output/chair_background_slow_depth/ --iteration 10000 --mode texture
python render.py --eval --model_path output/chair_background_slow_depth/ --iteration 13000 --mode texture
python render.py --eval --model_path output/chair_background_slow_depth/ --iteration 16000 --mode texture
python render.py --eval --model_path output/chair_background_slow_depth/ --iteration 20000 --mode texture
python render.py --eval --model_path output/chair_background_slow_depth/ --iteration 25000 --mode texture
python render.py --eval --model_path output/chair_background_slow_depth/ --iteration 30000 --mode texture
python render.py --eval --model_path output/chair_background_slow_depth/ --iteration 35000 --mode texture
python render.py --eval --model_path output/chair_background_slow_depth/ --iteration 40000 --mode texture
python render.py --eval --model_path output/chair_background_slow_depth/ --iteration 47000 --mode texture
python render.py --eval --model_path output/chair_background_slow_depth/ --iteration 55000 --mode texture
python render.py --eval --model_path output/chair_background_slow_depth/ --iteration 70000 --mode texture
python metrics.py --model_paths output/chair_background_slow_depth/