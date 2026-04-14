# Verkehrsbasierte Hub-Simulation in Zürich

## Datenquelle

Die zugrunde liegenden Verkehrsdaten stammen von der Stadt Zürich:

https://data.stadt-zuerich.ch/dataset/sid_dav_verkehrszaehlung_miv_od2031

Die Stadt Zürich stellt Verkehrsdaten zur Verfügung, bei denen **Messstellen die Anzahl Fahrzeuge pro Stunde zählen**.

Diese Daten erlauben es, die **Verkehrsbelastung im Stadtgebiet über den Tagesverlauf** zu analysieren und darauf basierend Simulationen durchzuführen.

---

# Beispielanwendung

## Problemstellung

Ein Logistikunternehmen möchte herausfinden:

- Bei welchen **zwei Hubs** Pakete gelagert werden sollen  
- **Wie viele Pakete** pro Hub benötigt werden  
- Von welchem Hub Kunden ihre Pakete am wahrscheinlichsten abholen  

Ziel ist es, die Pakete so zu verteilen, dass **Kunden ihre Pakete möglichst schnell erhalten**.

---

## Modellidee

Ein Kunde, der in Zürich wohnt, möchte sein Paket möglichst schnell erhalten.  
Dazu sucht er eine **optimale Route zu einem der verfügbaren Hubs**.

Die Route wird über ein Netzwerk aus **Verkehrsknoten (Messstellen)** berechnet.

Zwei Faktoren beeinflussen die Reisezeit:

- **Verkehrsbelastung an einem Knoten**  
- **Distanz zum nächsten Knoten**

---

## Lösungsansatz

Dieses Problem lässt sich gut mit **Markov-Ketten** modellieren.

Dabei wird ein **Graph aus Verkehrsknoten** aufgebaut:

- Knoten = Messstellen im Verkehrsnetz  
- Kanten = mögliche Übergänge zwischen Knoten  
- Übergangskosten = Funktion aus
  - Distanz
  - Verkehrsbelastung

Durch **Simulation zufälliger Kundenstandorte** kann bestimmt werden:

- Zu welchem Hub ein Kunde wahrscheinlich fährt  
- Wie viele Kundenanfragen jeder Hub erhält  

---

# Simulationsergebnis

Die Simulation zeigt, dass sich die **Anfragen pro Hub über den Tagesverlauf ändern**, da sich auch die **Verkehrsbelastung innerhalb der Stadt Zürich** verändert.

## Beispielvisualisierung

<img alt="Simulation der Hub-Auslastung" src="/home/gerj/Documents/playground/smartUMH/zur_sim/plots/2026-01-06/sim_map_2026-01-06_00:00:00.png" width="700"/>

*Abbildung 1: Simulation von Kundenanfragen und optimalen Routen zu Logistik-Hubs.*

<img alt="Hub-Auslastung über den Tag" src="/home/gerj/Documents/playground/smartUMH/zur_sim/plots/distribution.png" width="700"/>

*Abbildung 2: Kundenanfragen pro Logistik-Hubs über den Tag.*

---

# Dynamische Entwicklung über den Tag

Die Verkehrsbelastung verändert sich im Laufe des Tages. Dadurch verschieben sich auch die optimalen Routen und damit die Hub-Auslastung.

Eine Animation der Simulation über den gesamten Tagesverlauf ist hier verfügbar:
