import dxcam
from PIL import Image
camera = dxcam.create()  # returns a DXCamera instance on primary monitor
frame = camera.grab()
Image.fromarray(frame).show()
