import os
import sys
import numpy as np

# Aggiunge la cartella superiore (dove si trova data.py) ai percorsi di Python
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Punta alla cartella 'data' che si trova nel parent directory
data_root = os.path.join(parent_dir, 'data')
os.environ['PY123D_DATA_ROOT'] = data_root
os.environ['AV2_DATA_ROOT'] = data_root

from data import load_av2_dataset
def main():
    print("Initializing dataset...")
    # Load just 1 scene (log) to make it fast for testing
    ds = load_av2_dataset(data_root=data_root, split='av2-sensor_val', max_scenes=1)
    
    print(f"Successfully loaded dataset!")
    print(f"Total frames: {len(ds)}")
    print(f"Total scenes: {ds.num_scenes}")
    
    print("\nLoading first frame...")
    sample = ds[0]
    
    print("\n--- Frame 0 Details ---")
    print(f"Log Name: {sample.log_name}")
    print(f"Iteration: {sample.iteration}")
    print(f"Stereo Left Image: shape={sample.stereo_left_image.shape}, dtype={sample.stereo_left_image.dtype}")
    print(f"Stereo Right Image: shape={sample.stereo_right_image.shape}, dtype={sample.stereo_right_image.dtype}")
    print(f"LiDAR Points: shape={sample.lidar_points.shape}, dtype={sample.lidar_points.dtype}")
    
    depth_nz = np.count_nonzero(sample.lidar_depth_map)
    print(f"Depth Map: shape={sample.lidar_depth_map.shape}, non-zero valid pixels={depth_nz}")
    
    print(f"Ego Position (Global): {sample.ego_position}")
    print(f"Total 3D Detections: {len(sample.detections)}")
    
    # Check how many 3D detections are actually visible in the cameras (projected to 2D)
    visible_left = sum(1 for d in sample.detections if d.bbox_2d_left is not None)
    visible_right = sum(1 for d in sample.detections if d.bbox_2d_right is not None)
    print(f"Detections visible in Left Camera: {visible_left}")
    print(f"Detections visible in Right Camera: {visible_right}")
    
    if len(sample.detections) > 0:
        det = sample.detections[0]
        print(f"\n--- First Detection Details ---")
        print(f"Label: {det.bbox_3d.label}")
        print(f"3D Center (Global): {det.bbox_3d.center_xyz}")
        if det.bbox_2d_left:
            b = det.bbox_2d_left
            print(f"2D BBox (Left Camera): ({b.u_min:.1f}, {b.v_min:.1f}) to ({b.u_max:.1f}, {b.v_max:.1f})")

if __name__ == "__main__":
    main()
