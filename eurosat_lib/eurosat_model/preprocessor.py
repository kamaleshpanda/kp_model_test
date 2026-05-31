import numpy as np
import rasterio
from PIL import Image
import torchvision.transforms as transforms
import skimage.morphology as morph
from skimage.segmentation import slic
from skimage import graph

class MSPreprocessor:
    def preprocess(self, filepath):
        with rasterio.open(filepath) as src:
            img = src.read()
        
        img = img.astype(np.float32)
        for i in range(img.shape[0]):
            band_min, band_max = img[i].min(), img[i].max()
            if band_max > band_min:
                img[i] = (img[i] - band_min) / (band_max - band_min)
                
        # B02, B03, B04, B08 (Indices 1, 2, 3, 7)
        spatial = img[[1, 2, 3, 7], :, :]
        spectral = img.mean(axis=(1, 2))
        
        # Calculate NDVI for morphological processing: (B08 - B04) / (B08 + B04)
        b08 = img[7]
        b04 = img[3]
        ndvi = (b08 - b04) / (b08 + b04 + 1e-8)
        
        # Morphological operations
        struct_elem = morph.disk(3)
        opened = morph.opening(ndvi, struct_elem)
        closed = morph.closing(ndvi, struct_elem)
        
        morph_features = np.stack([opened, closed], axis=0)
        
        return {
            'spatial': spatial,
            'spectral': spectral,
            'morph': morph_features
        }

class RGBPreprocessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def preprocess(self, filepath):
        img = Image.open(filepath).convert('RGB')
        tensor = self.transform(img)
        
        # Extract graph features using SLIC and RAG
        img_np = np.array(img)
        segments = slic(img_np, n_segments=50, compactness=10, start_label=1)
        rag = graph.rag_mean_color(img_np, segments)
        
        num_nodes = len(rag.nodes)
        num_edges = len(rag.edges)
        if num_edges > 0:
            edge_weights = [d['weight'] for u, v, d in rag.edges(data=True)]
            mean_weight = float(np.mean(edge_weights))
            std_weight = float(np.std(edge_weights))
        else:
            mean_weight = 0.0
            std_weight = 0.0
            
        graph_features = np.array([num_nodes, num_edges, mean_weight, std_weight], dtype=np.float32)
        padded_graph = np.zeros(16, dtype=np.float32)
        padded_graph[:4] = graph_features
        
        return {
            'rgb': tensor,
            'graph': padded_graph
        }
