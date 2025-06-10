import os
import time

here = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(here, "images")
test_images = [
    os.path.join(image_dir, "dwpose_test_1_pexels_helena_lopes.jpg"),
    os.path.join(image_dir, "dwpose_test_2_pexels_elle_hughes.jpg"),
    os.path.join(image_dir, "dwpose_test_3_pexels_andres_piacquadio.jpg"),
    os.path.join(image_dir, "dwpose_test_4_pexels_jopwell.jpg"),
]   

def test_detector() -> None:
    from dwpose import DWPoseDetector
    from PIL import Image

    dwpose = DWPoseDetector.from_pretrained(device="cuda")
    assert dwpose is not None
    warmed_up = False

    for image_path in test_images:
        image = Image.open(image_path)
        if not warmed_up:
            results = dwpose(image)
            warmed_up = True

        start_time = time.perf_counter()
        results = dwpose(image)
        end_time = time.perf_counter()
        print(f"Processed {image_path} in {end_time - start_time:0.3f} seconds")
