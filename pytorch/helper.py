import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def load_dataset(path='./data', batch_size=64):
  """
  Loads the CIFAR10 and upsamples for model. Performs data augmentation on training set.
  
  :param path: path to data or where it will download.
  :param batch_size: number of examples in a batch
  """
  print("Loading the CIFAR10 dataset")

  transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(), # scale RGB 0-255 to 0-1
    # normalize with known mean and std deviation of the CIFAR10 dataset
    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
  ])

  train_transform = transforms.Compose([
    transforms.Resize(224),  # Resize before any augmentation
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
  ])

  # get training data
  train_dataset = datasets.CIFAR10(root=path, train=True, download=True, transform=train_transform)
  # get test data
  test_dataset = datasets.CIFAR10(root=path, train=False, download=True, transform=transform)
  # load the training data
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=8,pin_memory=True)
  # load the test data
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=8,pin_memory=True)

  print(f"Loaded train data: {len(train_loader.dataset)} total samples, {len(train_loader)} batches\n"
      f"Loaded test data: {len(test_loader.dataset)} total samples, {len(test_loader)} batches")

  return train_loader, test_loader



def plot_metrics(metrics):
  """
  Graphs the accuracy and loss from training
  
  :param metrics: output of train_model()
  """
  train_losses = metrics.get('train_loss',None)
  test_losses = metrics.get('test_loss',None)
  train_accs = metrics.get('train_acc',None)
  test_accs = metrics.get('test_acc',None)

  epochs = range(1, len(train_losses) + 1)

  plt.figure(figsize=(12, 5))

  # loss Graph
  plt.subplot(1, 2, 1)
  if train_losses:
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
  if test_losses:
    plt.plot(epochs, test_losses, label='Test Loss', marker='s')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Training vs Test Loss')
  plt.legend()
  plt.grid(True, linestyle='--', alpha=0.6)

  # accuracy Graph
  plt.subplot(1, 2, 2)
  if train_accs:
    plt.plot(epochs, train_accs, label='Train Accuracy', marker='o')
  if test_accs:
    plt.plot(epochs, test_accs, label='Test Accuracy', marker='s')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy (%)')
  plt.title('Training vs Test Accuracy')
  plt.legend()
  plt.grid(True, linestyle='--', alpha=0.6)

  plt.tight_layout()
  plt.show()

def train_model(model,train_loader,test_loader,train=True,test=True,device='cpu',epochs=10,lr=1e-3,weight_decay=1e-3,name="name",save=False,acc=80.0):
  """
  Main training loop using CrossEntropyLoss and Adam Optimizer
  
  :param model: SqueezeNetCIFAR10
  :param train_loader: output of load_dataset()
  :param test_loader: output of load_dataset()
  :param train: boolean to train
  :param test: boolean to test
  :param device: device to run on
  :param epochs: total number of epochs
  :param lr: learning rate
  :param weight_decay: weight decay for Adam
  :param name: name of file to save to (don't need .pth)
  :param save: boolean to save highest accuracy
  """
  model.to(device)
  metrics = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

  # TRAINING LOOP
  optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

  criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
  criterion_test = nn.CrossEntropyLoss()
  # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
  # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


  for e in range(epochs):
    print(f"Epoch [{e+1}/{epochs}] ",end='')
    if train:
      model.train()
      train_loss, total_examples, correct = 0.0, 0, 0

      for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True) # zero gradients
        outputs = model(inputs,e) # forward pass
        loss = criterion(outputs,labels) # get loss from cost function
        loss.backward() # backward propagation
        optimizer.step() # update gradients

        # train_loss += loss.item() # track total loss up to this point
        train_loss += loss.item() * labels.size(0)
        _, pred_ind = outputs.max(1) # get index of prediction (highest value)
        total_examples += labels.size(0) # update count for this epoch with batch size
        correct += pred_ind.eq(labels).sum().item() # return count of correct predictions

      train_loss /= total_examples # get average per example
      train_acc = 100.0 * correct / total_examples

      metrics["train_loss"].append(train_loss)
      metrics["train_acc"].append(train_acc)

      print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% ",end='')

      # VALIDATION/TEST
    if test:
      model.eval()
      test_loss, total_examples, correct = 0.0, 0, 0

      with torch.no_grad():
        for inputs, labels in test_loader:
          inputs, labels = inputs.to(device), labels.to(device)
          outputs = model(inputs,e) # forward pass
          loss = criterion_test(outputs,labels) # get loss from cost function
          test_loss += loss.item() * labels.size(0) # update loss
          _, pred_ind = outputs.max(1) # get index of prediction (highest value)
          total_examples += labels.size(0) # update count for this epoch with batch size
          correct += pred_ind.eq(labels).sum().item() # return count of correct predictions

      test_loss /= total_examples
      test_acc = 100.0 * correct / total_examples

      if (save) and (acc > test_acc > 75.0) and (e > 3):
        fname = name + f"_{e}.pth"
        model.save_model(fname)

      metrics["test_loss"].append(test_loss)
      metrics["test_acc"].append(test_acc)

      print(f"Test/Val Loss: {test_loss:.4f}, Test/Val Acc: {test_acc:.2f}%")

  return metrics

def evaluate(model, test_loader,device='cpu'):
  """
  Function to evaluate FP32 model
  
  :param model: SqueezeNetCIFAR10
  :param test_loader: output of load_dataset()
  :param device: device to run on
  """
  model.eval()
  model.to(device)
  correct, total = 0, 0

  with torch.no_grad():
      for images, labels in test_loader:
          images = images.to(device, non_blocking=True)
          labels = labels.to(device, non_blocking=True)
          outputs = model(images,epoch=None,inference=True)
          _, pred = outputs.max(1)
          correct += pred.eq(labels).sum().item()
          total += labels.size(0)

  acc = 100.0 * correct / total
  return acc

def verify_quantized_model(model, test_loader, device = 'cpu', total_bits=8, int_bits=4):
    """
    Verify that using quantized weights gives expected accuracy. Simulates what will happen in Vitis HLS.
    
    :param model: SqueezeNetCIFAR10 model
    :param test_loader: output of load_dataset()
    :param device: device to run on
    :param total_bits: total number of bits for fixed point
    :param int_bits: number of integer bits in fixed point
    """
    model.eval()
    frac_bits = total_bits - int_bits
    scale = 2 ** frac_bits
    qmin = -2 ** (int_bits - 1)
    qmax = (2 ** (int_bits - 1)) - 1
    
    # create a copy with frozen quantized weights
    quantized_state = {}
    for name, param in model.state_dict().items():
        if param.dtype.is_floating_point and 'weight' in name:
            w = param.cpu()
            w_int = torch.clamp(torch.round(w * scale), qmin * scale, qmax * scale)
            quantized_state[name] = (w_int / scale).to(device)
        else:
            quantized_state[name] = param
    
    # load quantized weights
    model.load_state_dict(quantized_state)
    model.to(device)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, epoch=None, inference=True)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy