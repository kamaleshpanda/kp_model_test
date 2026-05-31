import os
import torch
from .model import MultiModalFusionModel
from .preprocessor import MSPreprocessor, RGBPreprocessor

class eurosat_model:
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
            print("Downloading model weights (313 MB)... This will only happen once.")
            import urllib.request
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            url = "https://media.githubusercontent.com/media/kamaleshpanda/kp_model_test/refs/heads/main/eurosat_lib/eurosat_model/weights/best_eurosat_model_best.pth"
            urllib.request.urlretrieve(url, weights_path)
            print("Download complete!")
            
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        
        self.ms_prep = MSPreprocessor()
        self.rgb_prep = RGBPreprocessor()
        
        self.classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
                        'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
                        'River', 'SeaLake']

    def predict(self, ms_image_path, rgb_image_path):
        ms_data = self.ms_prep.preprocess(ms_image_path)
        rgb_data = self.rgb_prep.preprocess(rgb_image_path)
        
        spatial = torch.FloatTensor(ms_data['spatial']).unsqueeze(0).to(self.device)
        spectral = torch.FloatTensor(ms_data['spectral']).unsqueeze(0).to(self.device)
        rgb = rgb_data['rgb'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(spatial, spectral, rgb)
            _, predicted = torch.max(outputs, 1)
            
        return self.classes[predicted.item()]
