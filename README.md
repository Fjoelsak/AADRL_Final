
# Agentenbasierter Aktienhandel mittels Deep Reinforcement Learning

## Beschreibung
Dieses Repository enthält den Programmcode der Masterarbeit "Agentenbasierter Aktienhandel mittels  Deep Reinforcement Learning". 
Es ist zu beachten, dass dieses Repository alle Daten mit Ausnahme von Kaggle enthält: 
https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests

## Struktur

```bash 
Code/
│
├───data/                                   # Enthält die verwendeten Daten (unverarbeitet sowie verarbeitet)
│   ├───testing/                            # Testdatensatz (Zustandsbeschreibung & Turbulenz-Index)
│   │   ├───market_data.csv                                  
│   │   ├───turbulence_index.csv                                  
│   ├───training/                           # Trainingdatensatz (Zustandsbeschreibung & Turbulenz-Index)         
│   │   ├───market_data.csv                                  
│   │   ├───turbulence_index.csv                              
│   ├───validation/                         # Validierungsdatensatz (Zustandsbeschreibung & Turbulenz-Index)          
│   │   ├───market_data.csv                                  
│   │   ├───turbulence_index.csv                                  
│   ├───agent_environment_data.csv          # Daten für die Interaktion zwischen Agent & Umgebung (ergibt sich aus create_env.ipynb)
│   ├───market_data.csv                     # Marktdaten der einzelnen Aktien (stammen auf YahooFinance)
│   ├───sentiment_processed.csv             # Bestimmung der Stimmungseinschätzung für jeden Titel (Wahrscheinlichkeit des Sentiments)
│   ├───sentiment_processed_final.csv       # Nachverarbeitung der Stimmungseinschätzung (für jeden Titel nur positiv, negativ oder neutral)
│   ├───turbulence_index.csv                # Turbulenz-Index über den gesamten Datensatz
│
├───Ensemble_Strategy/                      # Code des Papers 'Deep reinforcement learning for automated stock trading: an ensemble strategy' (siehe https://github.com/AI4Finance-Foundation/FinRL-Trading/tree/master/old_repo_ensemble_strategy) (Nur die hinzugefügten Dateien werden nachfolgenden aufgezählt) 
│   ├───evaluation (own)/
│   │   ├───ensemble_strategy_results.csv   # Ergebnisse des Ensemble Agenten (mit 4 verschiedenen Startwerten des Zufallsgenerators trainiert)
│   │   ├───eval_final.ipynb                # Analyse zu der Leistung des trainierten Ageten 
│   ├───done_data.csv                       # Eigene Daten für die Interaktion zwischen Agent & Umgebung. Diese wurden angepasst, damit der Ensemble Agent diese einlesen kann
│   ├───results_seed_7                      # Ergebnisse, wenn der Agent mit Seed 7 trainiert wird
│   ├───results_seed_14                     # Ergebnisse, wenn der Agent mit Seed 14 trainiert wird
│   ├───results_seed_25                     # Ergebnisse, wenn der Agent mit Seed 25 trainiert wird
│   ├───results_seed_42                     # Ergebnisse, wenn der Agent mit Seed 42 trainiert wird
│             
├───env/                      
│   ├───init.py/           
│   ├───multistock_trading_v4.py            # Implementierung der Umgebung      
│
├───evaluate_agents/                      
│   ├───preprocess_data_for_ensemble/
│   │   ├───prepare_data_ensemble.ipynb     # Umformatierung der verwendeten Daten (eigene), um den Ensemble-Agenten darauf zu trainieren.
│   │   ├───prepared_data_ensemble.csv      # Ergebnisse der Umformatierung
│   ├───evaluate_agents.ipynb               # Analyse der Leistung und Strategie des Agenten & Vergleiche zu diversen Benchmarks
│
├───preprocessing/                      
│   ├───init.py           
│   ├───create_env.ipynb                    # Erzeugung der Daten für die Inteaktion zwischen Agent und Umgebung & Bestimmung des Turbulenz-Index 
│   ├───preprocessor.py                     # Klasse, um auf Basis der Marktdaten und Stimmungseinschätzungen einen gemeinsamen Datensatz zu erzeugen
│   ├───sentiment_analysis.ipynb            # Sentiment-Analyse, Downloaden der Marktdaten und zusätzliche Analysen zu den Datensätzen
│
├───train_agents/         
│   ├───training_eval/                      # Enthält für jeden der vier trainierten Agenten, die gespeicherten Modelle (für die verschiedenen Startwerte d. Zufallsgenerators / seeds)
│   ├───init.py           
│   ├───eval_callback.py                    # Callback, welcher für das Training des Agenten implementiert wurde (um besten Agenten zu speichern, etc.)
│   ├───train.ipynb                         # Training der Agenten mit verschiedenen Startwerten des Zufallsgenerators
│   ├───util_functions.py                   # Enthält 'hilfreiche' Funktionen, die beim Training der Agenten und der Analyse der Leistungen verwendet werden
│
├───.gitignore                              # Auflistung der Dateien und Ordner die von Git nicht getrackt werden sollen
├───environment.yml                         # Benötigte Bibliotheken
├───README.md                               # Projekt README Datei
└───requirements.txt                        # Benötigte Bibliotheken
```
