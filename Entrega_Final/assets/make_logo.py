"""
Genera un membrete institucional tipografico (wordmark) para la portada.
Composicion original en "Azul Tec" para uso academico en la caratula del
proyecto integrador; no reproduce el arte vectorial protegido de la marca.
"""
from PIL import Image, ImageDraw, ImageFont

AZUL = (0, 57, 166)          # Azul Tec aproximado
GRIS = (90, 96, 104)

W, H = 2200, 560
SC = 4  # supersampling para bordes nitidos
img = Image.new("RGBA", (W * SC, H * SC), (255, 255, 255, 0))
d = ImageDraw.Draw(img)

FONT_BOLD = "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"
FONT_REG = "/usr/share/fonts/truetype/freefont/FreeSans.ttf"

f_top = ImageFont.truetype(FONT_REG, 96 * SC)
f_main = ImageFont.truetype(FONT_BOLD, 200 * SC)

# Linea superior: "TECNOLOGICO DE"
top = "TECNOLÓGICO DE"
d.text((20 * SC, 40 * SC), top, font=f_top, fill=AZUL)

# Linea principal: "MONTERREY"
main = "MONTERREY"
d.text((16 * SC, 150 * SC), main, font=f_main, fill=AZUL)

# Barra de acento bajo el wordmark
bbox = d.textbbox((16 * SC, 150 * SC), main, font=f_main)
y_bar = bbox[3] + 30 * SC
d.rectangle([20 * SC, y_bar, bbox[2], y_bar + 16 * SC], fill=AZUL)

img = img.resize((W, H), Image.LANCZOS)
img.save("/home/user/Proyecto-Integrador/Entrega_Final/assets/logo_tec.png")
print("logo escrito")
