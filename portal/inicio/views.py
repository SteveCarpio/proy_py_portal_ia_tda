from django.shortcuts import render

# Create your views here.

from django.http import HttpResponse

def home(request):
    return HttpResponse("¡Hola, esta es mi primera web con Django!")



import pandas as pd

def formulario(request):
    nombre = None
    edad = None

    if request.method == 'POST':
        nombre = request.POST.get('nombre')
        edad = request.POST.get('edad')

    # Simulamos un DataFrame
    data = {
        'Nombre': ['Ana', 'Luis', 'Marta'],
        'Edad': [25, 32, 29]
    }
    df = pd.DataFrame(data)
    tabla_html = df.to_html(index=False)

    return render(request, 'inicio/formulario.html', {
        'nombre': nombre,
        'edad': edad,
        'tabla_html': tabla_html
    })
