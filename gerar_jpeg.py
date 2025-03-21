from PIL import Image

# Carregar as duas imagens
img1 = Image.open("imagem_one.jpeg")  # Substitua pelo nome correto da sua imagem
img2 = Image.open("imagem_two.jpeg")

# Escolher a orientação: 'horizontal' ou 'vertical'
orientacao = "horizontal"  # Mude para 'vertical' se quiser empilhar

# Definir tamanho da nova imagem
if orientacao == "horizontal":
    nova_largura = img1.width + img2.width
    nova_altura = max(img1.height, img2.height)
    nova_imagem = Image.new("RGB", (nova_largura, nova_altura))
    nova_imagem.paste(img1, (0, 0))
    nova_imagem.paste(img2, (img1.width, 0))
else:
    nova_largura = max(img1.width, img2.width)
    nova_altura = img1.height + img2.height
    nova_imagem = Image.new("RGB", (nova_largura, nova_altura))
    nova_imagem.paste(img1, (0, 0))
    nova_imagem.paste(img2, (0, img1.height))

# Salvar a imagem unificada
nova_imagem.save("imagem_final.jpeg", "JPEG")

# Mostrar a imagem final
nova_imagem.show()
