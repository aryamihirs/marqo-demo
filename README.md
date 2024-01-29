#### Multilingual Image Search Client with NFT Styled Bored Apes Dataset
This guide outlines the setup and execution of the `client.py` for searching within a dataset of 30 NFT-styled Bored Apes images using Marqo's Multilingual Image Search.

#### Prerequisites
* `Python 3.7` or higher
* `pip` (Python package manager)

#### Dataset
The dataset comprises 30 NFT-styled Bored Apes images. Each image is tagged with descriptions in both English and Spanish, catering to the multilingual search capabilities of Marqo.

#### Sample Queries
The script is pre-configured with the following sample queries in English and Spanish. Here are example queries to try.

* "A trendsetting primate with eye-catching accessories" / "Un primate pionero con accesorios llamativos"
* "A contemplative simian sporting urban fashion" / "Un simio contemplativo luciendo moda urbana"
* "A whimsical monkey with an artistic flair" / "Un mono caprichoso con un toque artístico"
* "A streetwise ape with a flair for the dramatic" / "Un simio callejero con un sentido dramático"
* "A cartoon ape showcasing a vibrant persona" / "Un simio de dibujos animados mostrando una personalidad vibrante"
* "A hip simian in casual chic attire" / "Un simio a la moda en atuendo casual chic"
* "An avant-garde monkey radiating cool vibes" / "Un mono vanguardista que irradia ondas geniales"
* "A dapper primate dressed to impress" / "Un primate elegante vestido para impresionar"

#### Modifying Sample Queries
To test different queries, modify the `query_text` parameter in the `perform_search` function call within `client.py`. This allows for experimentation with various search terms in both English and Spanish.

#### Setting Up and Running Marqo Server
* [Install Docker](https://docs.docker.com/get-docker/): Required for running the Marqo server.
* Run Marqo Server:

###### bash
```
docker pull marqo/marqo:latest
docker run -p 8882:8882 marqo/marqo:latest
```
This starts the Marqo server on http://localhost:8882

#### Environment Setup for Client
Clone Repository:

##### bash
```
git clone https://github.com/aryamihirs/marqo-demo
cd marqo-demo
```

#### Install Dependencies:
```
pip install marqo
```

#### Run Client:
```
python client.py
```

#### Code Walkthrough
Code Walkthrough
* `initialize_search_index(client_url, index_name, settings)`: Initializes Marqo search index.
  * `client_url`: URL for the Marqo server. Default is "http://localhost:8882".
  * `index_name`: Name of the search index, can be customized.
  * `settings`: Configures index settings like model and embedding normalization.

* `populate_index_with_images(mq, index_name, image_data)`: Populates the index with image data.
  * `mq.index(index_name).add_documents(...)`: Marqo method to add image data to the specified index.
  * `tensor_fields`: Specifies which fields are used for tensor operations, critical for the search algorithm.
  * `client_batch_size`: Determines the batch size for adding documents, affecting performance.

* `perform_search(mq, index_name, query_text, language_code)`: Executes a search.
  * `mq.index(index_name).search(...)`: Marqo method to perform the search.
  * `q`: The search query string.
  * `attributes_to_retrieve`: Specifies which attributes to retrieve, like URL and description.
  * `limit`: Limits the number of search results returned.
 
* `display_image_urls(search_results)`: Extracts and displays image URLs.
  * Iterates over search_results to print each image URL.

#### Notes
* Ensure the hardware requirements (multi-core CPU, CUDA-compatible GPU with at least 8GB memory) are met for optimal performance.
* Refer to the a more detailed article for  implementation and usage.
