
import sys
import os
import torch
# Add current directory to path so we can import Models
sys.path.append(os.getcwd())

from Models.LinkNet import LinkNet

def test_linknet():
    print("Initializing LinkNet...")
    try:
        model = LinkNet()
        print("Model initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    model.eval()
    # Mock input (Batch size 2, 3 channels, 512x512)
    x = torch.randn(2, 3, 512, 512)
    print(f"Running forward pass with input {x.shape}...")
    
    try:
        roads, orients = model(x)
        print("Forward pass successful.")
        
        # Check outputs
        print(f"Roads output list length: {len(roads)}")
        if len(roads) > 0:
            print(f"Roads[0] shape: {roads[0].shape}")
            
        print(f"Orients output list length: {len(orients)}")
        if len(orients) > 0:
            print(f"Orients[0] shape: {orients[0].shape}")
            
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_linknet()
