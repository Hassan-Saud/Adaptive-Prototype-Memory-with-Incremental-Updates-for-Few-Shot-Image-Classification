# ResNet Backbone
from torchdivision import models

class ResNetBackbone(nn.Module):
    def __init__(self, num_classes):
        super(ResNetBackbone, self).__init__()
        # Load the pretrained ResNet model
        self.resnet = resnet34(pretrained=True)
        #"""
        # Freeze the early layers
        for name, child in self.resnet.named_children():
            if name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                # Ensure later layers are trainable
                for param in child.parameters():
                    param.requires_grad = True #"""

        # Remove the fully connected layer
        self.resnet.fc = nn.Identity()
        
        # Placeholder for gradients and activations
        #self.gradients = None
        #self.activations = None
    
    #def activations_hook(self, grad):
        #self.gradients = grad
    
    def forward(self, x):
        # Forward pass through the ResNet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        # Save activations and register hook for gradients
        #self.activations = x  # Save activations from layer4
        #x.register_hook(self.activations_hook)  # Hook to get gradients
        #print(f"Activation shape after layer4: {x.shape}")
        
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x


# Inception Backbone
# Define the InceptionBackbone class
class InceptionBackbone(nn.Module):
    def __init__(self, num_classes):
        super(InceptionBackbone, self).__init__()
        
        # Load the pretrained InceptionV3 model
        self.inception = models.inception_v3(pretrained=True)
        self.num_classes = num_classes
        
        # Freeze all layers except the last block
        for name, param in self.inception.named_parameters():
            if "Mixed_7" not in name:  # Only unfreeze the last block (Mixed_7)
                param.requires_grad = False

        
        # Replace the classifier with a new one for the required number of classes
        self.inception.fc = nn.Identity()

        # Placeholders for gradients and activations (for Grad-CAM, if needed)
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        # Pass input through the convolutional feature extractor
        x = self.inception.Conv2d_1a_3x3(x)
        x = self.inception.Conv2d_2a_3x3(x)
        x = self.inception.Conv2d_2b_3x3(x)
        x = self.inception.maxpool1(x)
        x = self.inception.Conv2d_3b_1x1(x)
        x = self.inception.Conv2d_4a_3x3(x)
        x = self.inception.maxpool2(x)
        x = self.inception.Mixed_5b(x)
        x = self.inception.Mixed_5c(x)
        x = self.inception.Mixed_5d(x)
        x = self.inception.Mixed_6a(x)
        x = self.inception.Mixed_6b(x)
        x = self.inception.Mixed_6c(x)
        x = self.inception.Mixed_6d(x)
        x = self.inception.Mixed_6e(x)
        x = self.inception.Mixed_7a(x)
        x = self.inception.Mixed_7b(x)
        x = self.inception.Mixed_7c(x)
        
        # Save activations and register a hook to capture gradients (for Grad-CAM, if needed)
        self.activations = x
        x.register_hook(self.activations_hook)
        
        # Apply global average pooling
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))  # Reduces spatial dimensions to 1x1
        x = torch.flatten(x, 1)  # Flatten the tensor
        
        return x


#SqueezeNet Backbone

class SqueezeNetBackbone(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(SqueezeNetBackbone, self).__init__()
        
        # Load pretrained SqueezeNet
        self.squeezenet = models.squeezenet1_1(pretrained=True)  # Using 1.1 version (more stable)
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # Freeze all layers except the last fire block (fire9)
        for name, param in self.squeezenet.named_parameters():
            if "features.12" not in name:  # fire9 is features.12 in SqueezeNet1.1 and features.10 for 1_0
                param.requires_grad = False
       
        # New single FC layer (input channels = 512 for SqueezeNet1.1)
        self.final_fc = nn.Identity()

        # Placeholders for Grad-CAM (optional)
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        # Forward through feature extractor
        x = self.squeezenet.features(x)
        
        # Save activations for Grad-CAM (optional)
        self.activations = x
        x.register_hook(self.activations_hook)
        
        # Apply global average pooling
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))  # Reduces spatial dimensions to 1x1
        x = torch.flatten(x, 1)  # Flatten the tensor
        
        return x
		
		
#DenseNet Backbone

class DenseNetBackbone(nn.Module):
    def __init__(self, num_classes):
        super(DenseNetBackbone, self).__init__()
        global FEATURE_DIM
        # Load the pretrained DenseNet model (we'll use densenet161 here)
        self.densenet = models.densenet161(pretrained=True)
        self.num_classes = num_classes
        
        # Freeze all layers except the last dense block
        for name, param in self.densenet.named_parameters():
            if "denseblock4" not in name:  # Only unfreeze the last dense block
                param.requires_grad = False
        
        # Extract the input features of the final classifier layer
        FEATURE_DIM = self.densenet.classifier.in_features
        
        # Replace classifier with Identity since we only need features
        self.densenet.classifier = nn.Identity()
        
        # --- Grad-CAM placeholders ---
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        """Hook to save gradients for Grad-CAM."""
        self.gradients = grad

    def forward(self, x):
        # Pass input through DenseNet feature extractor
        x = self.densenet.features(x)
        
        # --- Grad-CAM logic (disabled) ---
        self.activations = x
        x.register_hook(self.activations_hook)
        
        # Apply global average pooling
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))  # Reduces spatial dims to 1x1
        x = torch.flatten(x, 1)  # Flatten
        
        return x


#VGG Backbone
class VGG16Backbone(nn.Module):
    def __init__(self, num_classes):
        super(VGG16Backbone, self).__init__()
        # Load the pretrained VGG16 model
        self.vgg = models.vgg19(pretrained=True)
        self.num_classes = num_classes
        
        # Freeze all layers except the last block (Block 5)
        for name, layer in self.vgg.features.named_children():
            # Freeze layers up to and including Block 4 (layer indices 0 to 23 for vgg16 and 26 for vgg19)
            if int(name) < 26:  # This will cover layers from 0 to Block 4
                for param in layer.parameters():
                    param.requires_grad = False
            else:  # Fine-tune Block 5
                for param in layer.parameters():
                    param.requires_grad = True
        
        
        # Replace the average pooling to perform global pooling (1x1 output)
        self.vgg.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Remove the fully connected classifier by setting it to an identity
        self.vgg.classifier = nn.Identity()
        
        # Placeholders for gradients and activations
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        # Pass input through the convolutional feature extractor
        x = self.vgg.features(x)
        
        # Save activations and register a hook to capture gradients
        self.activations = x
        x.register_hook(self.activations_hook)
        
        # Apply the global average pooling and flatten the output
        x = self.vgg.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x

