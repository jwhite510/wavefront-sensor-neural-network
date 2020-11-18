from django.shortcuts import render

# Create your views here.
def home(request):

    # files=get_react_build_files('editor')
    return render(request, "app/home.html")
