from pathlib import Path
from glob import glob
import os
import shutil
import numpy as np

def process_scene(image_folder:Path, max_num_images=200):
    name = image_folder.parts[-1]
    
    rgb_images = glob(str(image_folder / "rgb" / "*"))
    rgb_images = sorted(rgb_images)[::len(rgb_images)//(max_num_images) + 1]
    
    depth_images_names = [Path(x).parts[-1] for x in glob(str(image_folder / "depth" / "*"))]
    depth_image_times = np.array([float(x.split("-")[1].split(".")[0]) for x in depth_images_names])
    
    
    new_dir = Path("../data/redwood_proc/") / name 
    shutil.rmtree(new_dir, ignore_errors=True)
    
    new_image_dir = new_dir / "input"
    new_depth_dir = new_dir / "depth"
    
    new_image_dir.mkdir(exist_ok=True, parents=True)
    new_depth_dir.mkdir(exist_ok=True, parents=True)
    
    image_idx = 0
    for rgb_image in rgb_images:
        rgb_image = Path(rgb_image)
        rgb_time = float(rgb_image.parts[-1].split("-")[1].split(".")[0])
        closest_depth_idx = np.abs(depth_image_times-rgb_time).argmin()
        
        depth_image = image_folder/"depth"/ depth_images_names[closest_depth_idx]

        shutil.copy2(rgb_image, new_image_dir/f"{image_idx}.jpg")
        shutil.copy2(depth_image, new_depth_dir/f"{image_idx}.png")
        
        image_idx += 1
        print(rgb_image)
        print(depth_image)
    
if __name__=="__main__":
    process_scene(Path("../data/redwood/00002/"))
    