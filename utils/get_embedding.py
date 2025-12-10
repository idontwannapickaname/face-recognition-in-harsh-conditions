import torch
from torchvision import transforms
from face_alignment import align
from backbones import get_model
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)

def get_embedding(image_path: str, model=None) -> torch.Tensor:
    """
    Given the path to a face image, returns the embedding vector using the EdgeFace model.
    """
    if model is None:
        model_name="edgeface_xxs"
        model=get_model(model_name)
        checkpoint_path=f'checkpoints/{model_name}.pt'
        print(f'Loading model {model_name} from {checkpoint_path}...')
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f'Model {model_name} loaded.')
    model=model.to(device)
    model.eval()

    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    aligned = align.get_aligned_face(image_path) # align face
    if aligned:
        transformed_input = transform(aligned) # preprocessing
        print("Aligned face found.")
    else:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transformed_input = transform(img) # preprocessing
        print("No aligned face found, using original image.")
        
    transformed_input = transformed_input.unsqueeze(0).to(device)
    print("Transformed input shape:", transformed_input.shape)

    # extract embedding
    embedding = model(transformed_input)
    return embedding

  
if __name__ == "__main__":
    img_path = "data/normal/dung/1.jpg"
    embedding = get_embedding(img_path)
    print("Embedding shape:", embedding.shape)