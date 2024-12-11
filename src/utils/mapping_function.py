import torch

class MappingFunction:
    def __init__(self):
        pass

    def klobuchar(elevation_angle):
        """Klobuchar Mapping Function"""
        E = elevation_angle
        mapping_function = (96 - E) / 90
        return 1.0 + 2 * mapping_function ** 3

    def slm(E):
        """Single Layer Model Mapping Function"""
        R = 6371.0  # Radius of the Earth in km
        H = 450.0  # Height of the ionospheric shell in km
        mapping_function = torch.cos(torch.arcsin(R / (R + H) * torch.sin(torch.pi/2-E)))
        return 1.0 / mapping_function
    
    def mslm(E):
        """Modified Single Layer Model Mapping Function"""
        R = 6371.0  # Radius of the Earth in km
        H = 506.7  # Height of the ionospheric shell in km
        alpha = 0.9782
        mapping_function = torch.cos(torch.arcsin(R / (R + H) * torch.sin(alpha*(torch.pi/2-E))))
        return 1.0 / mapping_function
