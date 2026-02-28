import os
import random

# Paths
identity_file = './data/identity_CelebA.txt'
train_file = './data/trainset.txt'
test_file = './data/testset.txt'
gan_file = './data/ganset.txt'

def generate_splits():
    if not os.path.exists(identity_file):
        print(f"[ERROR] {identity_file} not found. Please place it in the ./data/ folder.")
        return

    # Read identities from the text file
    identity_map = {}
    with open(identity_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                img_name, ident = parts[0], parts[1]
                
                if ident not in identity_map:
                    identity_map[ident] = []
                identity_map[ident].append(img_name)

    # Sort identities by the number of images they have (descending)
    # This ensures the target model has plenty of training data per class
    sorted_identities = sorted(identity_map.keys(), key=lambda k: len(identity_map[k]), reverse=True)

    # 1. Private Set: First 1000 identities for the Target Classifier
    private_identities = sorted_identities[:1000]
    
    train_lines = []
    test_lines = []
    
    # Map original identity strings to new classification indices (0 to 999)
    for class_idx, ident in enumerate(private_identities):
        images = identity_map[ident]
        random.shuffle(images)
        
        # 80/20 train/test split for the classifier
        split_idx = int(len(images) * 0.8)
        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]
        
        for img in train_imgs:
            train_lines.append(f"{img} {class_idx}\n")
        for img in test_imgs:
            test_lines.append(f"{img} {class_idx}\n")

    # 2. Public Set (GAN): Next identities until we hit ~30,000 images
    gan_lines = []
    gan_image_count = 0
    
    for ident in sorted_identities[1000:]:
        if gan_image_count >= 30000:
            break
        for img in identity_map[ident]:
            if gan_image_count >= 30000:
                break
            gan_lines.append(f"{img}\n")
            gan_image_count += 1

    # Write the files out
    with open(train_file, 'w') as f:
        f.writelines(train_lines)
    with open(test_file, 'w') as f:
        f.writelines(test_lines)
    with open(gan_file, 'w') as f:
        f.writelines(gan_lines)

    print(f"[SUCCESS] Generated {train_file} with {len(train_lines)} images (1000 classes).")
    print(f"[SUCCESS] Generated {test_file} with {len(test_lines)} images (1000 classes).")
    print(f"[SUCCESS] Generated {gan_file} with {len(gan_lines)} images for the GAN.")

if __name__ == '__main__':
    # Fixed seed for reproducibility
    random.seed(42)
    generate_splits()