import math
from PIL import Image, ImageDraw
from PIL import ImagePath

side = 6
# xy = [
    # ((math.cos(th) + 1) * 90,
     # (math.sin(th) + 1) * 60)
    # for th in [i * (2 * math.pi) / side for i in range(side)]
    # ]

xy = [(1.0, 60.0), (3.0, 111), (200.0, 111.1),
        (0.0, 60.0), (4.9, 8.0), (135.0, 8.0)]

# xy = [(100.0, 200.0), (400.0, 200.0), (400.0, 400.0)]

image = ImagePath.Path(xy).getbbox()
size = list(map(int, map(math.ceil, image[2:])))

img = Image.new("RGB", size, "#f9f9f9")
img1 = ImageDraw.Draw(img)
img1.polygon(xy, fill ="#eeeeff", outline ="blue")

img.show()
