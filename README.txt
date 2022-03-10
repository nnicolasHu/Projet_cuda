Ce dossier est le rendu de projet de programmation parallèle avec Cuda.
Elève : Martin Duguey et Nicolas Hu.

Il est composé du code séquentiel ("sequentiel.cpp") fourni par M.Ivan Mary ainsi que les
fichiers de post-traitement des résultats (postresultat, script.conf).

Nous avons utilisé la clock avec "walltime.c" et le fichier Makefile est fourni pour 
prendre en charge le compilateur nvcc et copiler sur le noeud gpucreos1 du cluster.

Le code parallèlisé est dans le fichier "advectiondiffusion.cu".

Pour compiler le code : make all
Pour exécuter le code : make run 
Pour post-traiter les résultats et afficher la figure au temps T : make draw
Pour vérifier si le résultats correspond avec le fichier de référence : make verif

Le fichier de référence resRef.txt a été obtenu avec le code séquentiel pour T=7,5s.


