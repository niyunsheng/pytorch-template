from models.NYS import NYS
def get_model(modelname=None, class_num=1000):
    assert modelname
    if modelname == 'nys':
        return NYS(class_num=class_num)
    else:
        raise "not defined modelname"
