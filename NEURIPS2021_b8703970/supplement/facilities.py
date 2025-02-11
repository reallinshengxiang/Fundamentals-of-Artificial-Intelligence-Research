import pickle

# Store and retrieve objects


def Store(obj, name, path):
    f = open(path + name, 'wb')
    pickle.dump(obj, f)
    f.close()


def Retrieve(name, path):
    f = open(path + name, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj
