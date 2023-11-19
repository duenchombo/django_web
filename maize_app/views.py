from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage

from tensorflow.keras.applications import VGG16

conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
from keras.preprocessing import image
import numpy as np
from keras.models import load_model

saved_model = load_model("./model/maize_disease_classification_best.h5")
names = np.array(["HEALTHY", "MLN", "MSV"])


# Create your views here.
def homePageView(request):
    # return HttpResponse("Hello, World!")
    return render(request, "temp_redesigned.html", {"a": 1})


def image_predict(request):
    # print(request.POST["img_path"])
    # print(request.POST.dict())
    # print(request.FILES.dict())

    # print(request.FILES.get("img_path"))
    file_object = request.FILES.get("img_path")
    # print(file_object)
    fs = FileSystemStorage()
    file_name = fs.save(file_object.name, file_object)
    # print(file_name)
    file_url = fs.url(file_name)
    # print(file_url)
    test_path = "." + file_url

    img = image.load_img(test_path, target_size=(150, 150))
    img = image.img_to_array(img)
    img = img / 255
    img_display = img.copy()
    img = np.reshape(img, (1, 150, 150, 3))
    img_feature = conv_base.predict(img)
    img_feature.shape
    img_feature_reshape = np.reshape(img_feature, (1, 4 * 4 * 512))

    output = saved_model.predict(img_feature_reshape)
    index_max_output = np.argmax(output)
    # proba = round(output[0][0], 2)
    out = names[index_max_output]
    print(out)
    proba_round = round(output.max(), 2)
    context = {
        "file_url": file_url,
        "index_max_output": index_max_output,
        "output": output,
        "names": names,
        # "proba": proba,
        "proba_round": proba_round,
        "out": out,
    }

    return render(request, "test.html", context)
