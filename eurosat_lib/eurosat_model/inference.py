import os
import torch
from .model import MultiModalFusionModel
from .preprocessor import MSPreprocessor, RGBPreprocessor

class EuroSATPredictor:
    def __init__(self, weights_path=None, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.model = MultiModalFusionModel(num_classes=10)
        
        if weights_path is None:
            # Look for weights in the packaged directory
            base_dir = os.path.dirname(os.path.abspath(__file__))
            weights_path = os.path.join(base_dir, "weights", "best_eurosat_model_best.pth")
            
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found at {weights_path}. "
                                    "Please download them using git lfs or provide a valid weights_path.")
            
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        
        self.ms_prep = MSPreprocessor()
        self.rgb_prep = RGBPreprocessor()
        
        self.classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
                        'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
                        'River', 'SeaLake']

    def predict(self, ms_image_path, rgb_image_path):
        """
        Predicts the class of a EuroSAT image pair.
        
        Args:
            ms_image_path (str): Path to the multi-spectral (.tif) image.
            rgb_image_path (str): Path to the RGB (.jpg) image.
            
        Returns:
            str: The predicted class name.
        """
        ms_data = self.ms_prep.preprocess(ms_image_path)
        rgb_data = self.rgb_prep.preprocess(rgb_image_path)
        
        spatial = torch.FloatTensor(ms_data['spatial']).unsqueeze(0).to(self.device)
        spectral = torch.FloatTensor(ms_data['spectral']).unsqueeze(0).to(self.device)
        morph = torch.FloatTensor(ms_data['morph']).unsqueeze(0).to(self.device)
        rgb = rgb_data['rgb'].unsqueeze(0).to(self.device)
        graph = torch.FloatTensor(rgb_data['graph']).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(spatial, spectral, rgb, morph, graph)
            _, predicted = torch.max(outputs, 1)
            
        return self.classes[predicted.item()]
