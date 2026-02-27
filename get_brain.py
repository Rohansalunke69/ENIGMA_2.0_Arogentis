import urllib.request
import json
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

url = "https://raw.githubusercontent.com/anvaka/brain-mesh/master/data/brain.obj"
try:
    with urllib.request.urlopen(url) as response:
        content = response.read().decode('utf-8')
        vertices = []
        for line in content.split('\n'):
            if line.startswith('v '):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.extend([float(parts[1]), float(parts[2]), float(parts[3])])
        
        with open("frontend/public/brain_points.json", "w") as f:
            json.dump(vertices, f)
        print(f"Success! Extracted {len(vertices)//3} vertices.")
except Exception as e:
    print(f"Error: {e}")
