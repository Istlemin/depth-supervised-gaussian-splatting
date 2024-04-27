# python render.py --eval --model_path output/chair/ --iteration 3000
# python render.py --eval --model_path output/chair/ --iteration 5000
# python render.py --eval --model_path output/chair/ --iteration 8000
# python render.py --eval --model_path output/chair/ --iteration 10000
# python metrics.py --model_paths output/chair/

# python render.py --eval --model_path output/chair_low_gaussian/ --iteration 2000
# python render.py --eval --model_path output/chair_low_gaussian/ --iteration 5000
# python metrics.py --model_paths output/chair_low_gaussian/

# python render.py --eval --model_path output/chair_low_gaussian_depth/ --iteration 2000 --textured_render
# python render.py --eval --model_path output/chair_low_gaussian_depth/ --iteration 5000 --textured_render
# python metrics.py --model_paths output/chair_low_gaussian_depth/ --textured

# python render.py --eval --model_path output/chair_depth/ --iteration 5000 --textured_render
# python render.py --eval --model_path output/chair_depth/ --iteration 10000 --textured_render
# python render.py --eval --model_path output/chair_depth/ --iteration 18000 --textured_render
# python metrics.py --model_paths output/chair_depth/ --textured


python render.py --eval --model_path output/chair_low_gaussian_depth/ --iteration 2000 --textured_render --blend_mode alpha
python render.py --eval --model_path output/chair_low_gaussian_depth/ --iteration 5000 --textured_render --blend_mode alpha
python render.py --eval --model_path output/chair_low_gaussian_depth/ --iteration 2000 --textured_render --blend_mode scores_softmax
python render.py --eval --model_path output/chair_low_gaussian_depth/ --iteration 5000 --textured_render --blend_mode scores_softmax
python render.py --eval --model_path output/chair_low_gaussian_depth/ --iteration 2000 --textured_render --blend_mode scores
python render.py --eval --model_path output/chair_low_gaussian_depth/ --iteration 5000 --textured_render --blend_mode scores
python metrics.py --model_paths output/chair_low_gaussian_depth/ --textured

# python render.py --eval --model_path output/chair_depth/ --iteration 5000 --textured_render --blend_mode alpha
# python render.py --eval --model_path output/chair_depth/ --iteration 10000 --textured_render --blend_mode alpha
# python render.py --eval --model_path output/chair_depth/ --iteration 18000 --textured_render --blend_mode alpha
# python render.py --eval --model_path output/chair_depth/ --iteration 5000 --textured_render --blend_mode scores_softmax
# python render.py --eval --model_path output/chair_depth/ --iteration 10000 --textured_render --blend_mode scores_softmax
# python render.py --eval --model_path output/chair_depth/ --iteration 18000 --textured_render --blend_mode scores_softmax
# python render.py --eval --model_path output/chair_depth/ --iteration 5000 --textured_render --blend_mode scores
# python render.py --eval --model_path output/chair_depth/ --iteration 10000 --textured_render --blend_mode scores
# python render.py --eval --model_path output/chair_depth/ --iteration 18000 --textured_render --blend_mode scores
# python metrics.py --model_paths output/chair_depth/ --textured