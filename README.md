# Application of machine learning techniques for book genre classification

This project studies the use of different machine learning tools for book classification by genre and topic. Final project of SD201 - Mining of Large Datasets - Télécom Paris.

Using a collection of more than 30000 English books from the Project Gutenberg free digital library, we tried different machine learning techniques such as decision trees, SVM and neural networks to create an algorithm that could give an appropriate subject label to a text. To extract features from the books, we used different NLP techniques such as TF-IDF and Word2Vec. 

## Running the project

IMPORTANT: You can download the word embeddings from this link and save them in the embeddings folder so they can be directly loaded in the notebook. Else, you can execute the word embedder, but it takes a while. Save them with the name written below:

X_test.npy: https://drive.rezel.net/s/zx3cbecMikAoSsN
Y_test.npy: https://drive.rezel.net/s/dydkmD8Y2S2ps2c
X.npy: https://drive.rezel.net/s/d5yRPEa4PHQ3Es7
Y.npy: https://drive.rezel.net/s/nxMFT7tfJmMjcaT

## Repository structure

This is the source code of our project. It is divided in the following files:

1. `gutenberg.py` : It provides a class for working with the dataset. It downloads it and provides methods for getting the text of each book.
2. `data_exploration.ipynb` : Includes the work related to the study of the dataset.# Application of machine learning techniques for book genre classification
3. `word2vec.ipynb` : Includes all the work using features generated by the Word2Vec algorithm.
4. `w2vec_embeddings.py`: Python script for generating the vector embedding for each book as described in the report.
5. `embeddings/`: Because it takes a while to generate the book embeddings, they are already generated and saved in this folder as matrices.
6. `algos_comparison*.ipynb`: Includes the plotting of the different results obtained.
7. `tfidf.ipynb`: Includes the different algorithms using for mining the tf-idf features.
8. `tfidf-NeuralNetwork`: Includes the neural networks used for classifying using the tf-idf features.

## Credits

Members of the team:

    Rodrigo Calzada Haro
    Alex Elenter
    Thibaud Labat
    Guillermo Toyos Marfurt
