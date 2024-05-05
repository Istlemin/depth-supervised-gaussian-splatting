# python render.py --eval --model_path output/chair_background_slow/ --iteration 2000
# python render.py --eval --model_path output/chair_background_slow/ --iteration 3000
# python render.py --eval --model_path output/chair_background_slow/ --iteration 4000
# python render.py --eval --model_path output/chair_background_slow/ --iteration 5000
# python render.py --eval --model_path output/chair_background_slow/ --iteration 7000
# python render.py --eval --model_path output/chair_background_slow/ --iteration 10000
# python render.py --eval --model_path output/chair_background_slow/ --iteration 15000
# python render.py --eval --model_path output/chair_background_slow/ --iteration 20000
# python render.py --eval --model_path output/chair_background_slow/ --iteration 30000
# python render.py --eval --model_path output/chair_background_slow/ --iteration 39000
# python metrics.py --model_paths output/chair_background_slow/

python render.py --eval --model_path output/chair_background_slow_depth/ --iteration 2000 --mode texture
python render.py --eval --model_path output/chair_background_slow_depth/ --iteration 3000 --mode texture
python render.py --eval --model_path output/chair_background_slow_depth/ --iteration 4000 --mode texture
python render.py --eval --model_path output/chair_background_slow_depth/ --iteration 5000 --mode texture
python render.py --eval --model_path output/chair_background_slow_depth/ --iteration 7000 --mode texture
python render.py --eval --model_path output/chair_background_slow_depth/ --iteration 10000 --mode texture
python render.py --eval --model_path output/chair_background_slow_depth/ --iteration 15000 --mode texture
python render.py --eval --model_path output/chair_background_slow_depth/ --iteration 20000 --mode texture
python render.py --eval --model_path output/chair_background_slow_depth/ --iteration 30000 --mode texture
python render.py --eval --model_path output/chair_background_slow_depth/ --iteration 39000 --mode texture
python metrics.py --model_paths output/chair_background_slow_depth/