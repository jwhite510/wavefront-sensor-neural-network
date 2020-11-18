from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse, JsonResponse
import json
import numpy as np

# Create your views here.
def home(request):

    # files=get_react_build_files('editor')
    return render(request, "app/home.html")

@csrf_exempt
def retrieve(request):
    data=json.loads(request.body)
    print("data =>", data)
    diffraction=np.array([[1,2,3],[4,5,6],[1,2,3]])
    xintensity=np.array([[1,2,3],[4,5,6],[1,2,3]])
    xphase=np.array([[1,2,3],[4,5,6],[1,2,3]])
    return JsonResponse({
        'diffraction':diffraction.tolist(),
        'xintensity':xintensity.tolist(),
        'xphase':xphase.tolist(),
        })
