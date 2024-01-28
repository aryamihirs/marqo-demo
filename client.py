import marqo
def initialize_search_index(client_url, index_name, settings):
    """
    Initializes the Marqo search index with the given settings.
    If the index already exists, it returns the existing index.
    """
    mq = marqo.Client(url=client_url)
    try:
        mq.create_index(index_name, settings_dict=settings)
    except marqo.errors.MarqoWebError as e:
        if e.code == 'index_already_exists':
            print(f"Index {index_name} already exists. Skipping creation.")
        else:
            raise
    return mq

def display_image_urls(search_results):
    """
    Extracts and displays the image URLs from the search results.
    """
    for hit in search_results.get('hits', []):
        print(hit.get('url'))

def populate_index_with_images(mq, index_name, image_data):
    """
    Populates the index with image data.
    """
    return (mq.index(index_name).add_documents(image_data,
                           tensor_fields=['description', 'description_es'],
                           device="cpu",
                           client_batch_size=1))

def perform_search(mq, index_name, query_text, language_code):
    """
    Performs a search in the specified index based on the query text and language code.
    """
    search_field = "description_es" if language_code == "es" else "description"
    search_results = mq.index(index_name).search(
        q=query_text,
        attributes_to_retrieve=[search_field, "url"],
        limit=1
    )
    return search_results


def main():
    client_url = "http://localhost:8882"
    index_name = "multilingual-image-search"
    settings = {
        "treatUrlsAndPointersAsImages": True,
        "model": "multilingual-clip/XLM-Roberta-Large-Vit-L-14",
        "normalizeEmbeddings": True,
    }

    # Initialize Marqo client and index
    mq = initialize_search_index(client_url, index_name, settings)

    # Image data to populate the index with
    image_data = [
        {"url": "https://github.com/aryamihirs/marqo-demo/blob/main/data/File%201.jpg",
         "description": "A cartoonish ape with a sly expression wearing a pink fur coat against a turquoise background.",
         "description_es": "Un simio caricaturesco con expresión astuta llevando un abrigo de piel rosa sobre fondo turquesa."},
        {"url": "https://github.com/aryamihirs/marqo-demo/blob/main/data/File%202.jpg",
         "description": "A laid-back ape sporting a propeller cap and sunglasses, with a joint in its mouth, set against an orange backdrop.",
         "description_es": "Un simio relajado con gorra de hélice y gafas de sol, con un cigarrillo en la boca, sobre fondo naranja."},
        {"url": "https://github.com/aryamihirs/marqo-demo/blob/main/data/File%203.jpg",
         "description": "An edgy robotic ape with a dinosaur hat, green ooze, and a cybernetic arm against a yellow background.",
         "description_es": "Un simio robótico y atrevido con un sombrero de dinosaurio, baba verde y un brazo cibernético sobre fondo amarillo."},
        {"url": "https://github.com/aryamihirs/marqo-demo/blob/main/data/File%204.jpg",
         "description": "A cheerful ape with a red headband and a striped shirt, smiling against a light blue background.",
         "description_es": "Un simio alegre con una cinta roja en la cabeza y una camiseta a rayas, sonriendo sobre un fondo azul claro."},
        {"url": "https://github.com/aryamihirs/marqo-demo/blob/main/data/File%205.jpg",
         "description": "A grinning ape with a red baseball cap and an earring, displaying a bold set of teeth against a gray background.",
         "description_es": "Un simio sonriente con una gorra de béisbol roja y un arete, mostrando un conjunto de dientes audaces sobre un fondo gris."},
        {"url": "https://github.com/aryamihirs/marqo-demo/blob/main/data/File%206.jpg",
         "description": "A sophisticated ape in sunglasses and a branded 'HERMES PARIS' tee, smoking a cigar against an orange background.",
         "description_es": "Un simio sofisticado con gafas de sol y una camiseta de la marca 'HERMES PARIS', fumando un puro sobre un fondo naranja."},
        {"url": "https://github.com/aryamihirs/marqo-demo/blob/main/data/File%207.jpg",
         "description": "A young ape with a pacifier and a blue sleeping cap, looking sad against a light blue backdrop.",
         "description_es": "Un simio joven con un chupete y un gorro de dormir azul, luciendo triste sobre un fondo azul claro."},
        {"url": "https://github.com/aryamihirs/marqo-demo/blob/main/data/File%208.jpg",
         "description": "A regal ape with a golden crown and a melancholic expression against a blue background.",
         "description_es": "Un simio regio con una corona dorada y expresión melancólica sobre un fondo azul."},
        {"url": "https://github.com/aryamihirs/marqo-demo/blob/main/data/File%209.jpg",
         "description": "A contemplative white ape with blue eyes and a neutral expression against a white background.",
         "description_es": "Un simio blanco contemplativo con ojos azules y expresión neutra sobre un fondo blanco."},
        {"url": "https://github.com/aryamihirs/marqo-demo/blob/main/data/File%2010.jpg",
         "description": "A businesslike ape wearing a suit and a 'BAYC' cap, with a serious demeanor against a blue background.",
         "description_es": "Un simio de aspecto empresarial vistiendo un traje y una gorra 'BAYC', con un semblante serio sobre un fondo azul."},
        {"url": "https://github.com/aryamihirs/marqo-demo/blob/main/data/File%2011.jpg",
         "description": "An ape with laser eyes wearing a black T-shirt against an orange background.",
         "description_es": "Un simio con ojos láser usando una camiseta negra contra un fondo naranja."},
        {"url": "https://github.com/aryamihirs/marqo-demo/blob/main/data/File%2012.jpg",
         "description": "A skeptical ape wearing a dark shirt with a disapproving frown, set against a blue background.",
         "description_es": "Un simio escéptico con camisa oscura y gesto de desaprobación sobre fondo azul."},
        {"url": "https://github.com/aryamihirs/marqo-demo/blob/main/data/File%2013.jpg",
         "description": "A playful ape with green skin wearing a yellow shirt, sticking out its tongue at a small butterfly, against a blue circle.",
         "description_es": "Un simio juguetón de piel verde con camiseta amarilla, sacando la lengua a una pequeña mariposa, sobre un círculo azul."},
        {"url": "https://github.com/aryamihirs/marqo-demo/blob/main/data/File%2014.jpg",
         "description": "Two contrasting ape characters, one angelic with a halo and the other mischievous with laser eyes, against a divided backdrop.",
         "description_es": "Dos personajes simios contrastantes, uno angélico con halo y el otro travieso con ojos láser, contra un fondo dividido."},
        {"url": "https://github.com/aryamihirs/marqo-demo/blob/main/data/File%2015.jpg",
         "description": "A rainbow-hued, contemplative ape with a neutral expression against a plain background.",
         "description_es": "Un simio contemplativo con tonos arcoíris y expresión neutra contra un fondo liso."},
        {"url": "https://github.com/aryamihirs/marqo-demo/blob/main/data/File%2016.jpg",
         "description": "A content ape with sunglasses and a halo wearing a vibrant tropical shirt, against a purple background.",
         "description_es": "Un simio contento con gafas de sol y un halo llevando una camisa tropical vibrante, contra un fondo morado."},
        {"url": "https://github.com/aryamihirs/marqo-demo/blob/main/data/File%2017.jpg",
         "description": "A serious ape in a blue jacket with an 'M' on it, looking thoughtful against a dark teal background.",
         "description_es": "Un simio serio con una chaqueta azul con una 'M', pareciendo pensativo contra un fondo azul oscuro."},
        {"url": "https://github.com/aryamihirs/marqo-demo/blob/main/data/File%2018.jpg",
         "description": "A disgruntled ape with a baseball cap and glasses showing a subtle frown, against a green backdrop.",
         "description_es": "Un simio disgustado con una gorra de béisbol y gafas mostrando un ceño leve, contra un fondo verde."},
        {"url": "https://github.com/aryamihirs/marqo-demo/blob/main/data/File%2019.jpg",
         "description": "A laid-back ape with 3D glasses and a fez, casually smoking against an orange background.",
         "description_es": "Un simio relajado con gafas 3D y un fez, fumando casualmente contra un fondo naranja."},
        {"url": "https://github.com/aryamihirs/marqo-demo/blob/main/data/File%2020.jpg",
         "description": "An ape with a stern expression, wearing a hoodie with X-patterned eyes.",
         "description_es": "Un simio con expresión severa, llevando una sudadera con ojos en forma de X."},
        {"url": "https://github.com/aryamihirs/marqo-demo/blob/main/data/File%2021.jpg",
         "description": "A grinning ape with a cap and a black jacket, in front of bold 'NFT' letters.",
         "description_es": "Un simio sonriente con una gorra y chaqueta negra, frente a las letras 'NFT' en negrita."},
        {"url": "https://github.com/aryamihirs/marqo-demo/blob/main/data/File%2022.jpg",
         "description": "A curious ape with a contemplative gaze, set against a soft blue background.",
         "description_es": "Un simio curioso con mirada contemplativa, sobre un fondo azul suave."},
        {"url": "https://github.com/aryamihirs/marqo-demo/blob/main/data/File%2023.jpg",
         "description": "A portrait of an ape with a striped shirt, framed and observed by a person.",
         "description_es": "Un retrato de un simio con camisa a rayas, enmarcado y observado por una persona."},
        {"url": "https://github.com/aryamihirs/marqo-demo/blob/main/data/File%2024.jpg",
         "description": "A stern ape with a cap and plaid shirt, giving a serious look.",
         "description_es": "Un simio serio con gorra y camisa a cuadros, dando una mirada seria."},
        {"url": "https://github.com/aryamihirs/marqo-demo/blob/main/data/File%2025.jpg",
         "description": "A confident ape with a beret and sunglasses, exuding a cool aura.",
         "description_es": "Un simio confiado con boina y gafas de sol, exudando un aura genial."},
        {"url": "https://github.com/aryamihirs/marqo-demo/blob/main/data/File%2026.jpg",
         "description": "A naval officer ape with a captain's hat and red glasses, looking forward.",
         "description_es": "Un simio oficial naval con gorro de capitán y gafas rojas, mirando hacia adelante."},
        {"url": "https://github.com/aryamihirs/marqo-demo/blob/main/data/File%2027.jpg",
         "description": "A laughing ape with a top hat and open mouth, showing joy.",
         "description_es": "Un simio riendo con sombrero de copa y boca abierta, mostrando alegría."},
        {"url": "https://github.com/aryamihirs/marqo-demo/blob/main/data/File%2028.jpg",
         "description": "An artistic ape with a painter's hat and a thoughtful expression.",
         "description_es": "Un simio artístico con sombrero de pintor y expresión pensativa."},
        {"url": "https://github.com/aryamihirs/marqo-demo/blob/main/data/File%2029.jpg",
         "description": "A stylish ape on a digital device screen, sporting a yellow cap and black turtleneck.",
         "description_es": "Un simio estiloso en la pantalla de un dispositivo digital, con una gorra amarilla y un cuello de tortuga negro."},
        {"url": "https://github.com/aryamihirs/marqo-demo/blob/main/data/File%2030.jpg",
         "description": "A stylish ape with a yellow construction hat and black turtleneck, posing with attitude.",
         "description_es": "Un simio elegante con un casco de construcción amarillo y cuello alto negro, posando con actitud."}
    ]

    # Populate the index with images
    populate_index_with_images(mq, index_name, image_data)

    # Perform searches in different languages
    english_results = perform_search(mq, index_name, "A streetwise ape with a flair for the dramatic", "en")
    spanish_results = perform_search(mq, index_name, "Un simio callejero con un sentido dramático", "es")

    # Print search results
    print("Search results for English query:", english_results)
    print("Search results for Spanish query:", spanish_results)

    # Display image URLs from the search results
    print("Image URL from English query:")
    display_image_urls(english_results)

    print("\nImage URL from Spanish query:")
    display_image_urls(spanish_results)

if __name__ == '__main__':
    main()
