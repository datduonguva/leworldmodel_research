"""
This is to test the implementation of the ViT model on the classification task

"""
import numpy as np
from datasets import load_dataset
import torch
from vit import ViT, EMBEDDING_SIZE 

tiny_imagenet = load_dataset('Maysee/tiny-imagenet', split='train')

def data_generator(batch_size=4):
    """
    Generate a minibatch of data
    """

    while True:
        while True:
            batch_ids = np.random.randint(0, 100000, batch_size)
            images = [np.array(tiny_imagenet[i]['image']) for i in batch_ids]
            labels = np.array([np.array(tiny_imagenet[i]['label']) for i in batch_ids])
            if 2 in set(len(image.shape) for image in images):
                continue
            images = np.array(images)
            break
             

        yield images, labels

class MyClassificationModel(torch.nn.Module):
    """
    Create a model with a classification head
    """
    def __init__(self, n_classes=200, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes
        self.vit = ViT()
        self.classifier = torch.nn.Linear(
            in_features=EMBEDDING_SIZE,
            out_features=n_classes
        )

    def forward(self, x: torch.Tensor):
        features = self.vit(x)[:, 0]
        output = self.classifier(features)

        return output

        
if __name__ == '__main__':

    my_model = MyClassificationModel() 
    my_model.train()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(my_model.parameters(), lr=0.001)

    my_generator = data_generator()
    losses = []

    running_loss = 0
    for i, batch in enumerate(my_generator):
        images, labels = batch

        images = torch.Tensor(images).permute((0, 3, 1, 2))
        labels = torch.Tensor(labels).to(torch.long)

        optimizer.zero_grad()

        outputs = my_model(images)

        loss = loss_fn(outputs, labels) 
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if i % 10 == 9:
            last_loss = running_loss / 10 # loss per batch
            losses.append(last_loss) 
            print(losses[-3:])
            running_loss = 0
    
        if i == 2000:
            break

    assert losses[-1] < losses[0]
