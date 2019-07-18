# import dependencies
from mpi4py import MPI
from Model import SOM, read_dataset
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# Mendapatkan banyak step berdasarkan besar dimensi dan jumlah proses
def get_steps(size, dim_size):
  if size > dim_size:
    # Untuk proses yang rank nya lebih kecil dari besar dimensi akan diberikan step 1
    # Proses yang rank nya lebih besar dari dimensi tidak akan mengerjakan apapun sehingga diberikan step 0
    return [1 if i < dim_size else 0 for i in range(size)]
  elif (size % dim_size) != 0:
    # Setiap proses yang masih memiliki sisa dari hasil pembagian besar dimensi dengan jumlah proses
    # akan ditambahkan sisa pembagiannya terhadap beberapa proses agar jumlah dimensi yang dihitung
    # sesuai dengan input
    mod = dim_size % size
    div = dim_size // size
    return [div + 1 if i < mod else div for i in range(size)]
  else:
    # Jika besar dimensi habis dibagi jumlah proses maka setiap proses dapat memproses dimensi dengan besar yang merata
    div = dim_size // size
    return [div for i in range(size)]
  
# Driver
def run(row_size, col_size, epoch=500, start_time=0):
  # Buat COMM
  comm = MPI.COMM_WORLD

  # Dapatkan rank proses
  rank = comm.Get_rank()
  print("\n#running rank {}".format(rank))

  # Dapatkan total proses berjalan
  size = comm.Get_size()

  # ((1) Added) Ambil jumlah step berdasarkan ukuran baris matriksnya
  dim_steps = get_steps(size, row_size)
  print("Output layer for rank-{} = {} x {}".format(rank, dim_steps[rank], col_size))

  # Jalankan SOM untuk proses ini
  local_output_layer = SOM(dim_steps[rank], col_size, epoch)

  # Kumpulkan hasil seluruh SOM pada rank 0:
  if rank == 0:

    # Add this rank's value
    output_layer = local_output_layer
    print("#Rank {} Finished, now sending data...".format(rank))

    # Receive data from process with rank 1 until maximal rank
    for i in range(1, size):
      local_output_layer = comm.recv(source=i, tag=123)
      # print(local_output_layer)
      output_layer = np.concatenate((output_layer, local_output_layer))

    # Hitung waktu pemrosesan
    end_time = time.time()
    print("Waktu pemrosesan: {}".format(round(end_time-start_time, 2)))

    # Visualisasi hasil
    # Apabila melakukan visualisasi, maka window harus ditutup secara otomatis setelah beberapa selang waktu
    dataset = read_dataset()
    plt.scatter(x=dataset[:,0], y=dataset[:,1], color='blue')
    plt.scatter(x=output_layer[:,:,0], y=output_layer[:,:,1], color='red')
    plt.show(block=False)
    plt.pause(3)
    plt.close()

  # Jika bukan proses dengan rank 0, akan mengirimkan nilai proses ini ke proses dengan rank=0
  else:
    print("#Rank {} Finished, now sending data...".format(rank))
    comm.send(local_output_layer, dest=0, tag=123)

# ((2) Added) Method baru untuk mengecek apakah input
# yang diterima memiliki tipe integer atau long integer
def validate(input):
  if(isinstance(input, (int, long))):
    return True
  return False

if __name__ == '__main__':
  start = time.time()
  
  # ((3) Added) Sekarang input menerima ukuran baris dan kolom,
  # sehingga bisa diterima matriks bukan persegi (cth: 3 x 2)
  row, col, epoch = 6, 10, 25
  
  # ((4) Added) Lakukan validasi input baris, kolom, serta epoch
  if(validate(row) and validate(col) and validate(epoch)):
    print("Initial output layer size = {} x {}".format(row, col))
    run(row, col, epoch, start)
  else:
    print("Input yang anda masukkan tidak valid")
  