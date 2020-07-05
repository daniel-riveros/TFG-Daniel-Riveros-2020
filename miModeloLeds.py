## Lectura y ejecución del clasificador entre perros y gatos - By: Danie - ma. jun. 23 2020

# Imports necesarios
import sensor, image, lcd, time
import KPU as kpu
from fpioa_manager import fm
from board import board_info
from Maix import GPIO

# Definimos los GPIO a los pines para dar corriente a los leds
fm.register(board_info.PIN9, fm.fpioa.GPIO0)
led_perro=GPIO(GPIO.GPIO0,GPIO.OUT)
fm.register(board_info.PIN10, fm.fpioa.GPIO1)
led_gato=GPIO(GPIO.GPIO1,GPIO.OUT)

# Nos indica cual es el porcentaje mínimo que tiene que tener una clase para pertenecer
# realmente a ella, sino se marca como desconocido
THRESHOLD = 0.5

# Iniciamos el sensor (cámara) con la resolución de entrada para nuesta red
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((224, 224))
sensor.set_vflip(0)
sensor.set_hmirror(False)
sensor.run(1)


# Inicialización de la pantalla
lcd.init(freq=15000000)
lcd.clear()
lcd.draw_string(30,96,"Clasificador entre perros y gatos")
lcd.draw_string(60,112,"TFG Daniel Riveros Garcia")
lcd.draw_string(70,128,"ING. INFORMATICA HUELVA")
time.sleep(2)


# Etiquetas de nuestras clases

labels = ['Gato', 'Perro']

# Cargamos el modelo, en nuestro caso desde la memoria interna, aunque se podria tambien desde la sd
# task = kpu.load('/sd/model.kmodel')
task = kpu.load(0x100000)

clock = time.clock()

allImg = image.Image()

# Clase para encender los leds, apagamos ambos y encendemos unicamente el correspondiente a su clase
def lightUpLed(classIdx):

    led_perro.value(0)
    led_gato.value(0)

    if (classIdx == 0):
        led_perro.value(1)
        # LED PERRO = ROJO
    if (classIdx == 1):
        led_gato.value(1)
        # LED GATO = AZUL



# Bucle infinito para empezar con la clasificación entre perros y gatos
while(True):

    # Generamos una imagen con el sensor
    img = sensor.snapshot()
    clock.tick()
    # Tiempo de espera para bajar los fps (aprox 12 imágenes por segundo)
    time.sleep(1/12)
    # Generamos un array de probabilidades con el modelo y la imagen tomada
    fmap = kpu.forward(task, img)
    fps = clock.fps()

    # Obtenemos la máxima probabilidad
    plist = fmap[:]
    pmax = max(plist)

    # Generamos el texto con el porcentaje para ver a que clase pertenece (siempre que supere el
    # mínimo estimulado para pertecer a una clase.
    label_text = ""
    if pmax > THRESHOLD:
        max_index=plist.index(pmax)
        label_text = "{:.1f}%: {}".format(pmax*100, labels[max_index].strip())
    else:
        label_text = "Desconocido"

    # Pasamos la función para encender los leds
    lightUpLed(max_index)

    # Como la dimension de la imagen mostrada va a ser de 224x224 tenderemos un espacio a la derecha
    # para escribir.
    dim = 224

    # Mostramos imagen y un rectángulo negro para pintar encima
    allImg.draw_image(img, 0, 0)
    allImg.draw_rectangle(0, dim, 320, (240-dim), (0,0,0), 1, True)
    lcd.display(allImg, oft=(0, 0))

    # Mostramos debajo la clase a la que pertence
    lcd.draw_string(2, dim, label_text)

    # Mostramos a la derecha información sobre el TFG y los fps
    lcd.draw_string((dim + 10), 10, "TFG DANIEL")
    lcd.draw_string(dim+10, 30, "ETSI UHU")
    lcd.draw_string((dim + 10), 70, "FPS:")
    lcd.draw_string(dim+10, 90, "{0:.2f}".format(fps))


a = kpu.deinit(task)
