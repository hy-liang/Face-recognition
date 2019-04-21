from cnn_model import train_model, cnn_model1, cnn_model2, alex

if __name__ == '__main__':
    model = cnn_model2()
    train_model(model)