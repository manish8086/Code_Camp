import tensorflow_cloud as tfc

def scale_model():
    tfc.run(entry_point="mnist_example.py")

if __name__ == "__main__":
    scale_model()
