import mne
import numpy as np
import json
import os

print("Fetching fsaverage brain...")
try:
    fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
    pial_surf = os.path.join(fs_dir, 'surf', 'rh.pial')
    
    # Read the surface
    rr, tris = mne.read_surface(pial_surf)
    
    # Subsample points for performance in React
    # Keep 1 out of every 10 points
    sub_rr = rr[::10]
    
    # Scale points
    sub_rr = sub_rr * 100
    
    # Center points
    center = np.mean(sub_rr, axis=0)
    sub_rr = sub_rr - center
    
    # Convert to standard python list
    points_list = []
    # Both hemispheres (mirroring right to left for a full brain)
    for p in sub_rr:
        points_list.extend([float(p[0]), float(p[1]), float(p[2])])
    for p in sub_rr:
        points_list.extend([float(-p[0]), float(p[1]), float(p[2])])
        
    out_path = os.path.join("frontend", "public", "brain_points.json")
    with open(out_path, "w") as f:
        json.dump(points_list, f)
        
    print(f"Success! Saved {len(points_list)//3} points to {out_path}")
except Exception as e:
    print(f"Error: {e}")
