# ml_tag_prediction_api


Ce projet à pour but de de prédire les tags associé à des question github en fonction du contenus du tritre et du corps de texte.

Il comporte une API avec Flask, les fonctions de traitement du texte pour le rendre interprétable par le modèle, et le modèle déjà entrainé sous forme de fichier pkl.

Le fichier CountVerctorizer.pkl contient le vectoriser déjà 'fité' sur le jeu d'entrainement du modèle pour vectoriser le texte.

Le fichier PCA_TFIDF.pkl contient la PCA déjà 'fité' sur le jeu d'entrainement pour réduire les dimentions du texte afin de les rendre interprétable par le modèle.

Ce projet comporte aussi un Front, sous forme d'un simple fichier HTML en architecture monolitique pour plus de simplicitée.

La liste des librairies python necessaire au fonctionnement se trouvent dans le fichier requirements.txt