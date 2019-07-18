from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import math

# Normalize data
def normalize(data):
  xmin = min(data[:,0])
  xmax = max(data[:,0])
  ymin = min(data[:,1])
  ymax = max(data[:,1])

  data[:,0] = [(x-xmin)/(xmax-xmin) for x in data[:,0]]
  data[:,1] = [(y-ymin)/(ymax-ymin) for y in data[:,1]]
  
  return data

def read_dataset():
  dataset = np.genfromtxt('Dataset.csv', delimiter=',')
  dataset = normalize(dataset)
  return dataset

# Calculate euclidean distance
def euclidean(x, y):
    return ((x[0]-y[0])**2) + ((x[1]-y[1])**2)

# Neighborhood Function to calculate how close is two node
def neighborhood(dist, width):
    return math.exp(-(dist**2 / (2 * (width**2))))

# Update neighborhood
def update_neighborhood(width, iteration, time):
    return width * math.exp(-iteration / time)
     
# Update learning rate
def update_learning_rate(learning_rate, iteration, epoch):
    return learning_rate * math.exp(float(-iteration) / epoch)

def find_bmu(x, output_layer):
    # Find Best Matching Unit(BMU) of each input
    bmu = []
    min_dist = 999999

    for i in range(len(output_layer)):
        for j in range(len(output_layer)):
            curr_dist = euclidean(x, output_layer[i][j])
            if(curr_dist < min_dist):
                min_dist = curr_dist
                bmu = output_layer[i][j]

    return bmu

def SOM(row_size, col_size, epoch=500, width=2, learning_rate=0.1, new_learning_rate=0.1):

  data = read_dataset()
  # Initialization

  # Create random nxn matrices for output layer
  # Each cell have 2 random value [0,1]

  output_layer = np.random.random_sample(size=(row_size,col_size,2))

  # Set time constant
  time_w = epoch / math.log(width)

  indices = np.arange(len(data))

  for i in range(1, epoch+1):
      np.random.shuffle(indices)

      # Update Hyperparameter          
      width = update_neighborhood(width, i, time_w)
      new_learning_rate = update_learning_rate(learning_rate, i, epoch)

      # Iterasi seluruh isi dari dataset
      for idx in indices:
          x = data[idx]
          
          # Cari node pada output layer yang paling 'dekat' dengan x

          bmu = find_bmu(x, output_layer)

          # Update bobot seluruh node yang 'dekat' dengan BMU
          for row in range(row_size):
              for col in range(col_size):
                  curr_node = output_layer[row][col]
                  curr_dist = euclidean(bmu, curr_node)

                  # Apabila node ini berdekatan dengan BMU
                  if(curr_dist <= width**2):
                      output_layer[row][col] = curr_node + (new_learning_rate * neighborhood(curr_dist, width) * (x - curr_node))

  return output_layer

# SOM(10, 10, epoch=100)