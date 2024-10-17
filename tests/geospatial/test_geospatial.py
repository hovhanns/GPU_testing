# import cuspatial
import torch

def test_coordinate_transformation():
    # Define some simple coordinate data
    lonlat = torch.tensor([[120.0, 30.0], [121.0, 31.0]], device="cuda")
    
    # Perform a basic transformation
    result = cuspatial.geo_utils.haversine_distance(lonlat[:, 0], lonlat[:, 1], lonlat[:, 0] + 1, lonlat[:, 1] + 1)
    print(f"Coordinate transformation result: {result}")




