# Herkunft

Dieses Verzeichnis ist **kein eigener Code.** Es stammt aus
[microsoft/Mage](https://github.com/microsoft/Mage) — die Python-Bibliothek
`mage_flow`, die der Loader `mage_flow` in `serving/loaders.py` importiert.

Mitgeliefert wird sie, weil das Paket zum Zeitpunkt der Übernahme nicht auf PyPI
lag und das Serving-Image v2 sie zur Bauzeit braucht. `pyproject.toml` weist
Version `0.1.0` aus; der genaue Upstream-Commit wurde beim Kopieren nicht
festgehalten — wer aktualisiert, holt frisch aus dem Upstream und trägt den
Stand hier nach.

## Lizenz

**MIT**, siehe `LICENSE` in diesem Verzeichnis. Der Lizenztext samt
Copyright-Vermerk („Copyright (c) 2026 Microsoft") ist Bedingung der MIT-Lizenz
und muss bei jeder Kopie mitreisen — auch bei einem Fork dieses Repositories.

Er hat hier zunächst gefehlt. Das übrige Repository steht ebenfalls unter MIT,
die Lizenzen sind also verträglich; die Namensnennung war trotzdem geschuldet
und ist am 16.08.2026 nachgetragen worden.

## Nicht hier editieren

Änderungen an `mage_flow` gehören stromaufwärts. Was hier liegt, ist eine Kopie;
eine lokale Korrektur ginge beim nächsten Abgleich verloren und wäre nicht mehr
von fremdem Code zu unterscheiden. Anpassungen für dieses Repo gehören in
`serving/loaders.py`.
