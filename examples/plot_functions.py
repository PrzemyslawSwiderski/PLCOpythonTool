# Python program showing Graphical
# representation of tanh() function
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

in_array = np.linspace(-8, 8, 120)
out_array = sigmoid(in_array)

print("in_array : ", in_array)
print("\nout_array : ", out_array)

# red for numpy.tanh()
plt.figure(figsize=(5, 4))
plt.plot(in_array, out_array, color='blue')
plt.title("Funkcja logistyczna")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
#
# def ReLU(x):
#     return x * (x > 0)
#
# in_array = np.linspace(-8, 8, 120)
# out_array = ReLU(in_array)
#
# print("in_array : ", in_array)
# print("\nout_array : ", out_array)
#
# # red for numpy.tanh()
# plt.figure(figsize=(5, 4))
# plt.plot(in_array, out_array, color='blue')
# plt.title("Funkcja progowa unipolarna")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()


# in_array = np.linspace(-8, 8, 120)
# out_array = np.tanh(in_array)
#
# print("in_array : ", in_array)
# print("\nout_array : ", out_array)
#
# # red for numpy.tanh()
# plt.figure(figsize=(5, 4))
# plt.plot(in_array, out_array, color='blue')
# plt.title("Funkcja tangensa hiperbolicznego")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()


# in_array = np.linspace(-8, 8, 120)
# out_array = in_array
#
# print("in_array : ", in_array)
# print("\nout_array : ", out_array)

# # red for numpy.tanh()
# plt.figure(figsize=(5, 4))
# plt.plot(in_array, out_array, color='blue')
# plt.title("Funkcja liniowa")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()
