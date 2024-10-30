import torchvision as torchvision

try:
    import torch
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.rpn import AnchorGenerator
    from tqdm import tqdm
except ImportError as e:
    raise e


def create_model(num_classes):
    # Create a model using a pretrained backbone
    backbone = torchvision.models.resnet50(pretrained=True)
    backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))  # Remove the last layers
    backbone.out_channels = 2048  # ResNet50 output channels

    # Define an anchor generator
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),) * 5)

    # Create the Faster R-CNN model
    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator)

    return model


def train_model(dataloader, num_classes, device):
    model = create_model(num_classes).to(device)
    model.train()  # Set model to training mode

    # Define an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    num_epochs = 10  # Set the number of epochs

    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, targets in tqdm(dataloader):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            loss_dict = model(images, targets)

            # Calculate total loss
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass and optimize
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")
