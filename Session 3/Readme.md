# Backpropagation Calculations on Excel

## Write a neural network that can

1. take 2 inputs:
   1. an image from MNIST dataset, and
   2. a random number between 0 and 9
2. and gives two outputs:
   1. the "number" that was represented by the MNIST image, and
   2. the "sum" of this number with the random number that was generated and sent as the input to the network
      ![assign.png](./images/assign.png)   
3. you can mix fully connected layers and convolution layers
4. you can use one-hot encoding to represent the random number input as well as the "summed" output. 



## Input Data Preparation

```python
# Create a class to combine MNIST dataset and random numbers between 0 and 9
class Combined_Dataset():

  # We pass the train variable to get train or test data, and batch_size
  def __init__(self, train, batch_size):

      self.batch_size = batch_size
      # Load the MNIST data into the data_loader object
      self.data_loader = torch.utils.data.DataLoader(
          torchvision.datasets.MNIST('/files/', train=train, download=True,
                                transform=torchvision.transforms.Compose([
                                  torchvision.transforms.ToTensor(),
                                  torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
          batch_size=self.batch_size, shuffle=True)

      # Number of samples in the dataaset
      self.dataset = self.data_loader.dataset            

  # getitem function creats batches of our dataset on the fly by calling next(iter())
  def __getitem__(self, index):
      # Extract one batch of the MNIST data_loader
      image, label = next(iter(self.data_loader))

      # Generate randoms numbers between 0 and 9 of size=batch_size. The datatype is float as this is the input required for the network
      random_numbers = torch.tensor([randint(0,9) for _ in range(self.batch_size)], dtype=torch.float32)

      # Combine inputs and outputs as a list after transfering the tensors to the GPU
      x = [image.to(device), random_numbers.to(device)]
      # y labels for addition of number is reshaped to [32,1] as MSE requires it in this format
      y = [label.to(device), (label+random_numbers).reshape([32,1]).to(device)]

      return x, y

  def __len__(self):
      return len(self.data_loader)
```



## Model Architecture

```python
# Build the classifier and addition network
class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Classifier Network
        self.input1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3) # output size = 26
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3) # 24
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3) # 22
        self.pool = nn.MaxPool2d(2, 2) # 11

        # 1x1 convolution
        self.oneconv1 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1) # 11
        
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3) # 9
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3) # 7
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3) # 5
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=10, kernel_size=5) # 1

        # Addition network using fully connected layers
        self.input2 = nn.Linear(in_features=2, out_features=5)
        self.layer1 = nn.Linear(in_features=5, out_features=5)
        self.out2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, data1, data2):
        # Classifier Network forward prop
        # first block
        x = F.relu(self.input1(data1))
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.oneconv1(x))
        
        # second block
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        # third block
        x = self.conv6(x)
        output1 = torch.flatten(x, start_dim=1) # flatten all dimensions except batch      

        # Addition Network
        # Collect the output of the classifier network and select the index with maximum value
        x = torch.argmax(output1, dim=1)
        # Use torch.stack to create pairs of network outputs and random numbers
        x  = torch.stack((x.float(), data2), dim=1)
        
        # Pass the data through the addition network. No activation function required as addition of two numbers is a linear function
        x = self.input2(x)
        x = self.layer1(x)
        output2 = self.out2(x)

        # Return outputs from both the classifier and addition network
        return output1, output2
```



## Training the Model

```python
for epoch in range(10):  # Loop over the dataset multiple times

    total_loss = 0.0
    total_correct_1, total_correct_2 = 0, 0
    # Loop over the entire length of train data
    for i in range(len(train_data)):
        # Get the inputs and outputs
        # Input data x is a list of [images, random numbers], output data y is a list of [classes, sum of numbers]
        x, y = next(iter(train_data))

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + Backward + Optimize
        output1, output2 = model(x[0], x[1])
        # Use the CE loss for classification and MSE loss for addition 
        loss = CE_loss(output1, y[0]) + MSE_loss(output2, y[1])
        loss.backward()
        optimizer.step()

        # Calculate statistics
        total_loss += loss.item()
        total_correct_1 += output1.argmax(dim=1).eq(y[0]).sum().item()
        total_correct_2 += (torch.round(output2) == torch.round(y[1])).sum().item()
        
    # Print statistics        
    print(f"Epoch: {epoch+1}, loss: {total_loss}, Classification Acc: {100 * (total_correct_1/(len(train_data.dataset)))}, Addition Acc: {100 * (total_correct_2/(len(train_data.dataset)))}")
```



## Evaluating the Model

```python
correct_1, correct_2 = 0, 0
total_1, total_2 = 0, 0

# Since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    # Loop over the entire length of test data
    for i in range(len(test_data)):
        # Get the inputs and outputs
        # Input data x is a list of [images, random numbers], output data y is a list of [classes, sum of numbers]
        x, y = next(iter(test_data))

        # Calculate outputs by running data through the network 
        output1, output2 = model(x[0], x[1])

        # The class with the highest energy is what we choose as prediction
        _, predicted = torch.max(output1.data, 1)
        total_1 += y[0].size(0)
        # Calculate number of correction predictions for classifier
        correct_1 += (predicted == y[0]).sum().item()

        total_2 += y[1].to(device).size(0)
        # Calculate number of correction predictions for addition
        correct_2 += (torch.round(output2) == torch.round(y[1])).sum().item()

print('Accuracy of the network on the 10,000 test images: ', (100 * correct_1 / total_1))
print('Accuracy of the network on the 10,000 test images: ', (100 * correct_2 / total_2))
```

