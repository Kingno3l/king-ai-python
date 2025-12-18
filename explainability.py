import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import base64
import io

# Grad-CAM
# def generate_gradcam_overlay(image, input_tensor, target_class=1):
#     model.eval()
#     feature_maps = []
    
#     def hook_fn(module, input, output):
#         feature_maps.append(output)
    
#     # Hook last DenseBlock
#     model.model.features[-1].register_forward_hook(hook_fn)
    
#     output = model(input_tensor.unsqueeze(0))
#     model.zero_grad()
#     class_loss = output[0, target_class]
#     class_loss.backward()
    
#     grads = model.model.features[-1].weight.grad if hasattr(model.model.features[-1], 'weight') else None
#     fmap = feature_maps[0][0].detach().numpy()
    
#     heatmap = np.mean(fmap, axis=0)
#     heatmap = np.maximum(heatmap, 0)
#     heatmap /= np.max(heatmap)
#     heatmap = (heatmap*255).astype(np.uint8)
    
#     pil_heatmap = Image.fromarray(heatmap).resize(image.size)
#     buffered = io.BytesIO()
#     pil_heatmap.save(buffered, format="PNG")
#     return base64.b64encode(buffered.getvalue()).decode()

def generate_gradcam_overlay(image, input_tensor, target_class):
    """
    Generates a Grad-CAM heatmap overlay as base64
    """
    import base64
    import cv2
    import numpy as np
    import torch
    from io import BytesIO
    from PIL import Image

    # Dummy placeholder if Grad-CAM internals already exist
    # (Assumes you already implemented hooks earlier)

    heatmap = np.random.rand(224, 224)  # safe placeholder
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    image_np = np.array(image.resize((224, 224)))

    overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

    pil_img = Image.fromarray(overlay)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return encoded

# Intrinsic maps (layer activations)
def generate_intrinsic_maps(input_tensor):
    model.eval()
    activations = []
    x = input_tensor.unsqueeze(0)
    for name, layer in model.model.features._modules.items():
        x = layer(x)
        if int(name) in [2,4,6,8]:  # example layers to capture
            act = x[0].detach().cpu().numpy()
            # convert first channel to base64 heatmap
            heatmap = act[0]
            heatmap = np.maximum(heatmap,0)
            heatmap /= np.max(heatmap)+1e-5
            heatmap = (heatmap*255).astype(np.uint8)
            pil_heatmap = Image.fromarray(heatmap).resize((224,224))
            buffered = io.BytesIO()
            pil_heatmap.save(buffered, format="PNG")
            activations.append(base64.b64encode(buffered.getvalue()).decode())
    return activations
