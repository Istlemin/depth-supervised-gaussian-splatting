python render.py --eval --model_path output/chair_diffuse/ --iteration 2000
python render.py --eval --model_path output/chair_diffuse/ --iteration 5000
python render.py --eval --model_path output/chair_diffuse/ --iteration 10000
python metrics.py --model_paths output/chair_diffuse/

# python render.py --eval --model_path output/chair_diffuse_depth/ --iteration 5000 --textured_render --blend_mode alpha
# python render.py --eval --model_path output/chair_diffuse_depth/ --iteration 10000 --textured_render --blend_mode alpha
# python render.py --eval --model_path output/chair_diffuse_depth/ --iteration 5000 --textured_render --blend_mode scores_softmax
# python render.py --eval --model_path output/chair_diffuse_depth/ --iteration 10000 --textured_render --blend_mode scores_softmax
# python render.py --eval --model_path output/chair_diffuse_depth/ --iteration 5000 --textured_render --blend_mode scores
# python render.py --eval --model_path output/chair_diffuse_depth/ --iteration 10000 --textured_render --blend_mode scores
# python metrics.py --model_paths output/chair_diffuse_depth/ --textured