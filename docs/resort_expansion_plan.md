# Resort roster expansion plan

Written 27 Nov 2026. This is a plan only. No fetcher, validator or data file was edited. Updated the same day with a verification pass: coordinates for the Swiss and French rows, four new US East resorts, fixes for the doubtful bases, and a coordinate audit of the existing rows (section 1d).

## Headline numbers

- Current roster: 86 resorts (USA 22, Austria 10, Switzerland 11, France 20, Spain 13, Japan 10).
- Scouts returned 85 unique candidates. No candidate appeared twice across scouts. The verification pass added 4 more US East resorts (Gore Mountain, Stratton, Okemo, Mount Snow), so 89 candidates in all.
- Main roster after verification: 84 resorts (USA 24, Italy 18, Austria 14, Switzerland 8, France 14, Germany 5, Slovenia 1).
- Needs verification: 5 resorts. Peyragudes (France) still has only one base coordinate. Deer Valley, Killington, Northstar California (USA) and La Thuile (Italy) were moved out of the main roster because the audit found their coordinates do not match their base, and no corrected point could be sourced.
- Roster after the main list is added: 170 resorts (86 + 84).
- Open-Meteo calls per run: 31 today, about 55 after. Per day (4 runs): 124 today, about 220 after.
- Run time: 685 seconds today for 86 resorts. A straight-line estimate for 170 is about 1,354 seconds (23 minutes). The workflow timeout is 25 minutes, so the timeout must be raised in the first rollout phase (section 4).

## 1. Roster

### 1a. Main roster

Coordinates, base and summit come from the scout returns, corrected by the verification pass (section 1d). Confidence is the verifier's rating where a row was re-checked, otherwise the scout's. Every main-roster row has both coordinates and a summit, and every row is high or medium confidence. The Name column is the fetcher key; the full domain name is in the note where it differs.

**USA (24)**

| Name | Region | Lat, lon | Base m | Summit m | Confidence | Note |
|---|---|---|---|---|---|---|
| Stowe | Vermont | 44.53194, -72.7875 | 475 | 1340 | high | |
| Sugarbush | Vermont | 44.135098, -72.885962 | 452 | 1244 | high | |
| Stratton | Vermont | 43.11345, -72.90797 | 571 | 1181 | high | New (verification pass). Main Base Lodge. Official base 571 m looks low against the terrain model (about 647 m at the lodge); official figure kept. |
| Okemo | Vermont | 43.4023, -72.7173 | 346 | 1019 | high | New. Point is the main base (about 400 m); 346 m is the lowest lift at Jackson Gore, 2 km away. |
| Mount Snow | Vermont | 42.9679, -72.8944 | 579 | 1097 | high | New. Main Base Lodge. |
| Jay Peak | Vermont | 44.9389, -72.5061 | 553 | 1177 | high | Moved from 1b. Point is the Aerial Tram base. |
| Gore Mountain | New York | 43.6733, -74.007 | 304 | 1097 | high | New. Point is the main base lodge (about 465 m); 304 m is the lowest lift at North Creek Ski Bowl, 2.5 km east. |
| Whiteface | New York | 44.3536, -73.8614 | 372 | 1337 | high | Moved from 1b. Point is the base lodge, not the peak. Lift-served top used. |
| Loon Mountain | New Hampshire | 44.05492, -71.63496 | 260 | 930 | medium | Coordinates corrected (old point was the summit). Base 260 m from one source; the terrain model gives 296 m at the base village. |
| Sunday River | Maine | 44.4734, -70.8569 | 244 | 957 | high | |
| Sugarloaf | Maine | 45.05367, -70.30778 | 432 | 1291 | medium | Coordinates corrected (old point was the summit). Point is the base lodge (about 536 m); 432 m is the lowest lift. |
| Steamboat | Colorado | 40.459, -106.803 | 2103 | 3221 | high | |
| Winter Park | Colorado | 39.88694, -105.7625 | 2743 | 3676 | high | |
| Copper Mountain | Colorado | 39.50167, -106.15639 | 2960 | 3792 | high | |
| Keystone | Colorado | 39.605, -105.95417 | 2829 | 3782 | high | |
| Arapahoe Basin | Colorado | 39.64167, -105.87167 | 3286 | 3978 | high | Base sources differ by about 80 m. |
| Crested Butte | Colorado | 38.89972, -106.965 | 2858 | 3707 | high | |
| Snowbasin | Utah | 41.216, -111.857 | 1948 | 2850 | high | |
| Solitude | Utah | 40.615139, -111.588917 | 2435 | 3197 | high | |
| Brighton | Utah | 40.6, -111.58333 | 2669 | 3277 | high | |
| Powder Mountain | Utah | 41.38, -111.78028 | 2103 | 2713 | medium | Lift-served top used, not the 9,422 ft hike-to point. The resort is upside-down: the point is the top village (about 2,500 m) and 2,103 m is the lowest lift far below. |
| Taos Ski Valley | New Mexico | 36.596, -105.4545 | 2804 | 3804 | medium | Coordinates corrected (old point was about 500 m above base). New point is an administrative point, not a named lodge. |
| Alyeska | Alaska | 60.9705, -149.0982 | 76 | 1201 | high | |
| Schweitzer | Idaho | 48.367, -116.623 | 1207 | 1951 | high | Base corrected from 1,219 to 1,207 m (Wikipedia, Stella chair). Point is the main village at about 1,433 m. |

Deer Valley, Killington and Northstar California were removed from this table and moved to 1b (see section 1d).

**Italy (18)**

| Name | Region | Lat, lon | Base m | Summit m | Confidence | Note |
|---|---|---|---|---|---|---|
| Cortina d'Ampezzo | Veneto | 46.54028, 12.13611 | 1224 | 2930 | high | |
| Val Gardena | South Tyrol | 46.55, 11.767 | 1563 | 2518 | medium | |
| Alta Badia | South Tyrol | 46.55, 11.867 | 1568 | 2778 | medium | |
| Val di Fassa | Trentino | 46.475, 11.773 | 1450 | 2485 | medium | |
| Arabba-Marmolada | Veneto | 46.49718, 11.87409 | 1602 | 3269 | medium | Coordinates corrected (old point was 6 km east). Base changed from 1,645 to 1,602 m to match the village. |
| San Martino di Castrozza | Trentino | 46.267, 11.8 | 1487 | 2357 | medium | |
| Sestriere | Piedmont | 44.95845, 6.87869 | 2035 | 2800 | high | Coordinates moved to the village point. |
| Breuil-Cervinia | Aosta Valley | 45.9344, 7.6311 | 2050 | 3480 | medium | |
| Courmayeur | Aosta Valley | 45.783, 6.967 | 1224 | 3466 | medium | |
| Monterosa Ski | Aosta Valley | 45.83164, 7.726 | 1560 | 3275 | high | Coordinates moved to Champoluc. |
| Livigno | Lombardy | 46.533, 10.133 | 1816 | 2798 | medium | |
| Bormio | Lombardy | 46.467, 10.367 | 1225 | 3017 | high | |
| Madonna di Campiglio | Trentino | 46.23194, 10.82611 | 1522 | 2504 | high | |
| Passo del Tonale | Trentino | 46.25806, 10.58083 | 1883 | 3000 | medium | |
| Solda-Sulden | South Tyrol | 46.51667, 10.6 | 1906 | 3250 | medium | |
| Schnalstal | South Tyrol | 46.75651, 10.78275 | 2011 | 3212 | high | Coordinates moved to Kurzras (old point was down-valley at about 1,480 m). |
| Alpe di Siusi | South Tyrol | 46.54027, 11.61791 | 1850 | 2245 | high | Base corrected from 1,060 m (the Seis valley gondola) to 1,850 m at Compatsch; coordinates moved there. Summit not re-checked (skiresort says 2,220 m). |
| Kronplatz | South Tyrol | 46.77666, 11.95029 | 973 | 2275 | medium | Moved from 1b (Plan de Corones). Point is Reischach. |

La Thuile was removed from this table and moved to 1b (see section 1d).

**Austria (14)**

| Name | Region | Lat, lon | Base m | Summit m | Confidence | Note |
|---|---|---|---|---|---|---|
| Saalbach-Hinterglemm | Salzburg | 47.3904, 12.637 | 1003 | 2096 | high | Base corrected from 830 to 1,003 m (830 m is the Skicircus-wide low point in Leogang). |
| Mayrhofen | Tyrol | 47.167, 11.867 | 630 | 2500 | high | |
| Zillertal Arena | Tyrol | 47.2336, 11.88199 | 580 | 2500 | high | Coordinates moved to Zell am Ziller (old point was on the mountain at about 2,040 m). |
| Hintertux Glacier | Tyrol | 47.11502, 11.6825 | 1500 | 3250 | high | Coordinates moved to Hintertux village. |
| Stubai Glacier | Tyrol | 47.0118, 11.15619 | 1750 | 3210 | high | Moved from 1b. Point is the Mutterberg valley station. |
| Kitzsteinhorn | Salzburg | 47.2725, 12.75944 | 768 | 3029 | high | |
| Serfaus-Fiss-Ladis | Tyrol | 47.03861, 10.60583 | 1429 | 2828 | high | Base corrected from 1,200 to 1,429 m (1,200 m is the Ladis valley station). |
| SkiWelt Wilder Kaiser-Brixental | Tyrol | 47.5139858, 12.2092381 | 620 | 1869 | high | |
| Silvretta Montafon | Vorarlberg | 47.08021, 9.91971 | 700 | 2430 | high | Coordinates moved to Schruns. |
| Hochkönig | Salzburg | 47.40533, 12.90271 | 800 | 1900 | high | Coordinates moved to Maria Alm (old point was near the peak). |
| Bad Gastein | Salzburg | 47.17031, 13.10606 | 860 | 2300 | high | Coordinates moved to Bad Hofgastein, which matches the 860 m base. |
| Damüls-Mellau | Vorarlberg | 47.34998, 9.88119 | 690 | 2050 | high | Coordinates moved from Damüls to Mellau, which matches the 690 m base. |
| Ski Juwel Alpbachtal Wildschönau | Tyrol | 47.396, 11.91 | 830 | 2030 | medium | |
| Nassfeld | Carinthia | 46.6129, 13.27823 | 600 | 2002 | high | Coordinates moved to Tröpolach (old point was at the pass, about 1,550 m). |

**Switzerland (8)**

| Name | Region | Lat, lon | Base m | Summit m | Confidence | Note |
|---|---|---|---|---|---|---|
| Grindelwald-Wengen | Bern | 46.62453, 8.03254 | 1034 | 2472 | medium | Point is Grindelwald village. Summit sources disagree (2,400 to 2,500 m); 2,472 m is the Lauberhorn. Excludes Grindelwald-First. |
| Crans-Montana | Valais | 46.31194, 7.4825 | 1500 | 2927 | medium | Coordinate sources mark different halves of the twin resort, 1.2 km apart. |
| Adelboden-Lenk | Bern | 46.49306, 7.55861 | 1350 | 2200 | medium | Summit uncertain: 2,362 m if the whole lift pass is counted. |
| Gstaad | Bern | 46.47216, 7.28875 | 1050 | 2151 | medium | Gstaad (Saanenland). One precise coordinate source. Excludes Glacier 3000. |
| Villars-Gryon-Les Diablerets | Vaud | 46.29776, 7.05533 | 1253 | 2120 | medium | Excludes Glacier 3000. |
| Champéry | Valais | 46.17505, 6.87137 | 1050 | 2277 | medium | Portes du Soleil, Swiss side. Summit is the Swiss-side top (Pointe des Mossettes). |
| Scuol | Graubünden | 46.797, 10.2995 | 1250 | 2785 | high | Scuol-Motta Naluns. |
| Disentis | Graubünden | 46.70417, 8.85498 | 1130 | 2833 | medium | Base is the village; the lowest lift is at 1,227 m. |

**France (14)**

| Name | Region | Lat, lon | Base m | Summit m | Confidence | Note |
|---|---|---|---|---|---|---|
| La Rosière | Northern Alps | 45.6286, 6.84784 | 1850 | 2800 | medium | Espace San Bernardo. Shares the domain with La Thuile (held in 1b). |
| Valloire | Northern Alps | 45.16528, 6.42833 | 1430 | 2750 | medium | Valloire-Valmeinier (Galibier-Thabor). |
| Le Grand Massif (Les Carroz) | Northern Alps | 46.025, 6.64361 | 1140 | 2480 | medium | Flaine is already covered, so the point is Les Carroz. |
| Valmorel | Northern Alps | 45.46167, 6.44333 | 1400 | 2550 | medium | One coordinate source. |
| Les Contamines-Montjoie | Northern Alps | 45.82167, 6.72917 | 1164 | 2487 | medium | Summit sources range from 2,437 to 2,500 m. |
| Montgenèvre | Southern Alps | 44.93083, 6.72111 | 1860 | 2648 | high | Via Lattea; summit is the French sector top. |
| Puy-Saint-Vincent | Southern Alps | 44.83, 6.48722 | 1400 | 2750 | medium | Base is the 1400 village. |
| Les Orres | Southern Alps | 44.49301, 6.5571 | 1650 | 2720 | medium | Point is the 1650 station, not the old village. |
| La Grave-La Meije | Southern Alps | 45.04611, 6.30583 | 1450 | 3200 | medium | Freeride only. Summit is the cable car top. |
| Orcières-Merlette | Southern Alps | 44.6968, 6.32186 | 1850 | 2725 | high | Orcières-Merlette 1850. Point is the 1850 station. |
| Saint-Lary-Soulan | Pyrenees | 42.8125, 0.2944 | 1700 | 2515 | high | Point is the Pla d'Adet station, not the valley village. |
| Grand Tourmalet | Pyrenees | 42.91, 0.17917 | 1800 | 2500 | medium | La Mongie-Barèges. Point is La Mongie. |
| Ax 3 Domaines | Pyrenees | 42.70192, 1.81626 | 1400 | 2400 | medium | Point is the Bonascre plateau. |
| Font-Romeu-Pyrénées 2000 | Pyrenees | 42.52104, 2.05967 | 1777 | 2213 | medium | Point is the Pyrénées 2000 base. One precise coordinate source. |

France uses "Pyrenees" as a new region name in FRANCE_SKI_AREAS, matching the Spain list. Peyragudes stays in 1b.

**Germany (5)**

| Name | Region | Lat, lon | Base m | Summit m | Confidence | Note |
|---|---|---|---|---|---|---|
| Oberstdorf-Kleinwalsertal | Bavaria | 47.417, 10.283 | 820 | 2224 | high | |
| Garmisch Classic | Bavaria | 47.471, 11.064 | 700 | 2050 | high | |
| Zugspitze | Bavaria | 47.4134, 10.98 | 2000 | 2650 | medium | Moved from 1b. Point is the plateau hub (about 2,576 m); 2,000 m is the lowest lift 1.5 km east. |
| Feldberg | Baden-Württemberg | 47.8624, 8.0364 | 950 | 1450 | medium | Moved from 1b. 14 lifts. Point is the Seebuck base (about 1,280 m); 950 m is at Fahl, 1.5 km southwest (47.8561, 8.0049). |
| Winterberg | North Rhine-Westphalia | 51.2, 8.517 | 630 | 820 | medium | |

**Slovenia (1)**

| Name | Region | Lat, lon | Base m | Summit m | Confidence | Note |
|---|---|---|---|---|---|---|
| Kranjska Gora | Gorenjska | 46.485, 13.779 | 807 | 1282 | high | Passes the size bar on lift count only (20 km of pistes). |

Count check: 24 USA + 18 Italy + 14 Austria + 8 Switzerland + 14 France + 5 Germany + 1 Slovenia = 84.

### 1b. Needs verification (5)

These stay out of the fetcher until they are fixed.

| Name | Country | Why it is held |
|---|---|---|
| Peyragudes | France | Confidence low. Elevations are confirmed (1,600-2,400 m), but the resort has two 1,600 m bases (Les Agudes and Peyresourde) and only one source gives a base coordinate (Les Agudes 42.78838, 0.47676). Wikidata's point is 1.5 km away, mid-domain. |
| Deer Valley | USA | Confidence low. The old coordinates sit about 480 m above the 1,990 m base. 1,990 m is not Snow Park Lodge (about 2,200 m); it is probably the new Deer Valley East base, whose coordinates could not be sourced. Interim option: Snow Park Lodge (40.63729, -111.47833) with base 2,200 m. |
| Killington | USA | Confidence low. The coordinates are the Snowshed main base (about 677 m), but the 355 m base is the Skyeship base on Route 4, whose coordinates were not sourced. Either set the base to match Snowshed or source Skyeship coordinates. |
| Northstar California | USA | Confidence low. The old coordinates sit about 140 m above the base. A village point was estimated but not confirmed by any source. |
| La Thuile | Italy | Confidence low. The old coordinates sit about 350 m above the 1,450 m base. OpenStreetMap gives only an administrative centre (45.69715, 6.91829); no village or lift-base point was sourced. |

Count check: 1 France + 3 USA + 1 Italy = 5. The other 28 rows from the first list (Jay Peak, Whiteface, Kronplatz, Stubai, Zugspitze, Feldberg, 8 Swiss and 14 French) moved to 1a.

### 1c. Scout errors and doubts

I checked 12 rows against rough reference points (a town or village near each resort) using a haversine distance in python. All 12 landed within 1 to 24 km of their reference, and base was below summit on all of them. The reference points came from memory, so this tests placement, not precision. The checks did not catch the base problems below.

Base elevations that look wrong or inconsistent (each one is now resolved; see section 1d for the evidence):

1. **Saalbach-Hinterglemm, base 830 m.** The same scout says the village sits at 1,003 m (Wikipedia). One of those two numbers is wrong. Confirm before adding. *Resolved: 1,003 m. The 830 m figure is the lowest lift of the whole Skicircus, in Leogang and Fieberbrunn.*
2. **Alpe di Siusi, base 1,060 m.** That is the valley village. The plateau where the lifts run is well above that, close to 1,800 m (from memory). The base looks wrong for the ski area. Confirm before adding. *Resolved: 1,850 m at Compatsch, with the coordinates moved there.*
3. **Nassfeld, base 600 m.** That is the Tröpolach valley. The main ski area base is around 1,500 m (from memory). Confirm before adding. *Resolved: 600 m is a real base (the Tröpolach gondola). The coordinates were moved to Tröpolach to match it.*
4. **Serfaus-Fiss-Ladis, base 1,200 m.** The same scout cites the Serfaus village at 1,429 m. Confirm which base is meant. *Resolved: 1,429 m. 1,200 m is the Ladis valley station.*
5. **La Grave, base 1,470 m.** Snow-forecast gives 1,400 m. This is minor. *Resolved: 1,450 m, the official cable car base.*

Other notes:

- Kronplatz is marked low confidence but has coordinates. The scout's own rule moves it to verification, and I did so. *The verification pass raised it to medium (summit confirmed by two sources, base agrees with the terrain model at Reischach), so it is now in 1a.*
- Several scout rows (Jay Peak, Whiteface, Stubai, the Swiss and French rows) lack coordinates. The scouts did not fill them in. *All of them now have coordinates (section 1d).*
- Mixed base convention: some scouts used village heights (Val Gardena, Alta Badia) and some used lift bases. The fetcher's existing Chamonix entry uses the town base, so village heights are acceptable. Keep it consistent per resort.
- Overlaps with covered resorts, all kept:
  - Kitzsteinhorn sits inside the Zell am See-Kaprun lift pass, where Zell am See is covered.
  - Le Grand Massif includes Flaine (covered).
  - Val Gardena and Alpe di Siusi share a linked pass (about 181 km combined).
  - Monterosa and Champoluc are one area.
  - Passo del Tonale and Madonna di Campiglio are separate.

Coverage gaps in the scout returns:

- New York (Adirondacks) was listed as swept, but only Whiteface came back. Gore Mountain is missing. *Gore Mountain added in the verification pass.*
- Vermont was listed as swept, but Stratton, Okemo, Mount Snow, Bromley, Mad River Glen and Cannon (NH) are missing. *Stratton, Okemo and Mount Snow added. Bromley, Mad River Glen and Cannon are still missing.*
- Italy was not swept for Friuli, Plose, Carezza, Folgaria, Piancavallo, Pila, Sauze d'Oulx, Santa Caterina, Obereggen or Ahrntal (the scout's own list).
- Austria, Germany and Switzerland were swept by region, not by resort list. Styria and Upper Austria are not covered.

### 1d. Verification pass (27 Nov 2026)

A second pass checked coordinates against OpenStreetMap and Wikidata, and elevations against official resort sites, Wikipedia, skiresort.com and bergfex. It also checked each existing row's coordinates against a terrain model (EU-DEM 25 m in Europe, USGS 10 m in the US) to catch points that sit far above or below the stated base. The full source line for each row is in the RESORT_ELEVATIONS comments (section 2e).

Rules used: high or medium confidence with coordinates goes into 1a. Low confidence stays in 1b. Where the coordinates and the base describe different spots (a lodge point with the lowest-lift base), the row is kept if both numbers are sourced, and the gap is written in the note.

**Rows added to 1a (34).**

- From 1b (28): Jay Peak, Whiteface, Kronplatz, Stubai Glacier, Zugspitze, Feldberg, the 8 Swiss rows, and 14 of the 15 French rows.
- New candidates (4): Gore Mountain, Stratton, Okemo, Mount Snow.
- Lift-count question for Feldberg settled: 14 lifts on both the official site and skiresort.com. The 50+ figure is the wider lift pass.

**Base corrections on existing rows (5).**

| Resort | Old base | New base | Why |
|---|---|---|---|
| Saalbach-Hinterglemm | 830 | 1003 | 830 m is the Skicircus-wide low point, far from the village point. |
| Alpe di Siusi | 1060 | 1850 | 1,060 m is the Seis valley gondola; the plateau lift base is Compatsch. |
| Serfaus-Fiss-Ladis | 1200 | 1429 | 1,200 m is the Ladis valley station. |
| Arabba-Marmolada | 1645 | 1602 | Terrain model at the village gives 1,603 m. One source, so medium. |
| Schweitzer | 1219 | 1207 | Wikipedia gives 1,207 m for the lowest chair. |

Kronplatz kept the scout's 973 m base when it moved in; it agrees with the terrain model at Reischach (954 m).

**Coordinate corrections on existing rows (15).** The old point was checked against the terrain model, and the new point is the village or base that matches the stated base.

| Resort | Old point | New point | Problem with the old point |
|---|---|---|---|
| Zillertal Arena | 47.259306, 12.00194 | 47.2336, 11.88199 (Zell am Ziller) | On the mountain at about 2,040 m |
| Hochkönig | 47.420806, 13.063167 | 47.40533, 12.90271 (Maria Alm) | Near the 2,941 m peak |
| Bad Gastein | 47.16667, 13.2 | 47.17031, 13.10606 (Bad Hofgastein) | Mountainside at about 2,000 m |
| Silvretta Montafon | 46.9809, 9.9658 | 47.08021, 9.91971 (Schruns) | On the mountain at about 2,160 m |
| Damüls-Mellau | 47.2801, 9.89 | 47.34998, 9.88119 (Mellau) | Damüls village, about 720 m above the 690 m base |
| Hintertux Glacier | 47.11667, 11.66667 | 47.11502, 11.6825 (Hintertux) | About 440 m above base |
| Nassfeld | 46.5758, 13.2619 | 46.6129, 13.27823 (Tröpolach) | At the pass, about 950 m above the 600 m base |
| Schnalstal | 46.72222, 10.86083 | 46.75651, 10.78275 (Kurzras) | Down the valley at about 1,480 m |
| Sestriere | 44.95, 6.883 | 44.95845, 6.87869 (village) | About 160 m above base |
| Monterosa Ski | 45.833, 7.733 | 45.83164, 7.726 (Champoluc) | About 125 m above base |
| Arabba-Marmolada | 46.483, 11.95 | 46.49718, 11.87409 (Arabba) | 6 km east, down the valley |
| Alpe di Siusi | 46.54111, 11.6447194 | 46.54027, 11.61791 (Compatsch) | Plateau centre, not the lift base |
| Loon Mountain | 44.03611, -71.62167 | 44.05492, -71.63496 (base village) | The summit |
| Sugarloaf | 45.03139, -70.31306 | 45.05367, -70.30778 (base lodge) | The summit |
| Taos Ski Valley | 36.59, -105.43694 | 36.596, -105.4545 (village) | About 500 m above base |

Alpe di Siusi and Arabba appear in both tables, so 18 existing rows were corrected in all (5 base fixes, 15 coordinate fixes, 2 overlapping).

**Rows moved from 1a to 1b (4).** Deer Valley, Killington, Northstar California and La Thuile. The audit found their coordinates do not match their base, and no corrected point could be sourced. Reasons are in 1b.

**Confirmed, no change.** Powder Mountain is flagged but correct: it is an upside-down resort, with the village at the top. Its confidence is lowered to medium so the gap is visible.

## 2. Fetcher-ready entries

Each block goes in the file named above it. Stations are empty lists at launch. Station IDs have not been verified (see section 3). Build_resorts already adds the "no station" note for empty US lists. For the other countries, no note is added yet (see 3f).

Timezone: nothing to add. The fetcher asks Open-Meteo for `timezone=auto`, and the output picks up the zone (for example `Europe/Zurich` for Zermatt).

Region names follow the existing style. US uses state names. Austria uses state names (Tyrol, Salzburg, Vorarlberg, Carinthia). Italy uses Italian regions. Germany uses German states. Slovenia uses the Slovenian region name, Gorenjska, as Spain uses range names.

### 2a. US_SKI_AREAS (scripts/fetch_data_v2.py, around line 126)

```python
    # Vermont (no SNOTEL in the East; see plan section 3d)
    ("Stowe", 44.53194, -72.7875, "Vermont", 475, []),
    ("Sugarbush", 44.135098, -72.885962, "Vermont", 452, []),
    ("Stratton", 43.11345, -72.90797, "Vermont", 571, []),
    ("Okemo", 43.4023, -72.7173, "Vermont", 346, []),
    ("Mount Snow", 42.9679, -72.8944, "Vermont", 579, []),
    ("Jay Peak", 44.9389, -72.5061, "Vermont", 553, []),
    # New York
    ("Gore Mountain", 43.6733, -74.007, "New York", 304, []),
    ("Whiteface", 44.3536, -73.8614, "New York", 372, []),
    # New Hampshire
    ("Loon Mountain", 44.05492, -71.63496, "New Hampshire", 260, []),  # coords fixed (1d)
    # Maine
    ("Sunday River", 44.4734, -70.8569, "Maine", 244, []),
    ("Sugarloaf", 45.05367, -70.30778, "Maine", 432, []),  # coords fixed (1d)
    # Colorado (AWDB station lookup needed per resort)
    ("Steamboat", 40.459, -106.803, "Colorado", 2103, []),
    ("Winter Park", 39.88694, -105.7625, "Colorado", 2743, []),
    ("Copper Mountain", 39.50167, -106.15639, "Colorado", 2960, []),
    ("Keystone", 39.605, -105.95417, "Colorado", 2829, []),
    ("Arapahoe Basin", 39.64167, -105.87167, "Colorado", 3286, []),
    ("Crested Butte", 38.89972, -106.965, "Colorado", 2858, []),
    # Utah (AWDB station lookup needed per resort)
    ("Snowbasin", 41.216, -111.857, "Utah", 1948, []),
    ("Solitude", 40.615139, -111.588917, "Utah", 2435, []),
    ("Brighton", 40.6, -111.58333, "Utah", 2669, []),
    ("Powder Mountain", 41.38, -111.78028, "Utah", 2103, []),  # top-village resort (1d)
    # New Mexico, Alaska, Idaho (AWDB lookup needed; Alaska may have none)
    ("Taos Ski Valley", 36.596, -105.4545, "New Mexico", 2804, []),  # coords fixed (1d)
    ("Alyeska", 60.9705, -149.0982, "Alaska", 76, []),
    ("Schweitzer", 48.367, -116.623, "Idaho", 1207, []),  # base fixed (1d)
    # Held in 1b, not added: Deer Valley, Killington, Northstar California
```

### 2b. ITALY_SKI_AREAS (new list, after AUSTRIA_SKI_AREAS)

Same tuple shape as the Austria list. Stations are empty until a source is wired (see 3e).

```python
# =============================================================================
# ITALY - no station source wired yet (plan 3e). Stations empty at launch.
# Format: (name, lat, lon, region, station_elevation_m, stations)
# =============================================================================
ITALY_SKI_AREAS = [
    ("Cortina d'Ampezzo", 46.54028, 12.13611, "Veneto", 1224, []),
    ("Val Gardena", 46.55, 11.767, "South Tyrol", 1563, []),
    ("Alta Badia", 46.55, 11.867, "South Tyrol", 1568, []),
    ("Val di Fassa", 46.475, 11.773, "Trentino", 1450, []),
    ("Arabba-Marmolada", 46.49718, 11.87409, "Veneto", 1602, []),  # coords and base fixed (1d)
    ("San Martino di Castrozza", 46.267, 11.8, "Trentino", 1487, []),
    ("Sestriere", 44.95845, 6.87869, "Piedmont", 2035, []),  # coords fixed (1d)
    ("Breuil-Cervinia", 45.9344, 7.6311, "Aosta Valley", 2050, []),
    ("Courmayeur", 45.783, 6.967, "Aosta Valley", 1224, []),
    ("Monterosa Ski", 45.83164, 7.726, "Aosta Valley", 1560, []),  # coords fixed (1d)
    ("Livigno", 46.533, 10.133, "Lombardy", 1816, []),
    ("Bormio", 46.467, 10.367, "Lombardy", 1225, []),
    ("Madonna di Campiglio", 46.23194, 10.82611, "Trentino", 1522, []),
    ("Passo del Tonale", 46.25806, 10.58083, "Trentino", 1883, []),
    ("Solda-Sulden", 46.51667, 10.6, "South Tyrol", 1906, []),
    ("Schnalstal", 46.75651, 10.78275, "South Tyrol", 2011, []),  # coords fixed (1d)
    ("Alpe di Siusi", 46.54027, 11.61791, "South Tyrol", 1850, []),  # Compatsch; coords and base fixed (1d)
    ("Kronplatz", 46.77666, 11.95029, "South Tyrol", 973, []),  # Reischach base
    # Held in 1b, not added: La Thuile
]

# GERMANY and SLOVENIA: new lists, same shape. No station source wired (plan 3e).
GERMANY_SKI_AREAS = [
    ("Oberstdorf-Kleinwalsertal", 47.417, 10.283, "Bavaria", 820, []),
    ("Garmisch Classic", 47.471, 11.064, "Bavaria", 700, []),
    ("Zugspitze", 47.4134, 10.98, "Bavaria", 2000, []),  # plateau hub; 2,000 m is the lowest lift
    ("Feldberg", 47.8624, 8.0364, "Baden-Württemberg", 950, []),  # Seebuck point; 950 m is at Fahl
    ("Winterberg", 51.2, 8.517, "North Rhine-Westphalia", 630, []),
]
SLOVENIA_SKI_AREAS = [
    ("Kranjska Gora", 46.485, 13.779, "Gorenjska", 807, []),
]
```

### 2c. AUSTRIA_SKI_AREAS (add to the existing list, around line 225)

```python
    ("Saalbach-Hinterglemm", 47.3904, 12.637, "Salzburg", 1003, []),  # base fixed (1d)
    ("Mayrhofen", 47.167, 11.867, "Tyrol", 630, []),
    ("Zillertal Arena", 47.2336, 11.88199, "Tyrol", 580, []),  # Zell am Ziller; coords fixed (1d)
    ("Hintertux Glacier", 47.11502, 11.6825, "Tyrol", 1500, []),  # coords fixed (1d)
    ("Stubai Glacier", 47.0118, 11.15619, "Tyrol", 1750, []),  # Mutterberg valley station
    ("Kitzsteinhorn", 47.2725, 12.75944, "Salzburg", 768, []),
    ("Serfaus-Fiss-Ladis", 47.03861, 10.60583, "Tyrol", 1429, []),  # base fixed (1d)
    ("SkiWelt Wilder Kaiser-Brixental", 47.5139858, 12.2092381, "Tyrol", 620, []),
    ("Silvretta Montafon", 47.08021, 9.91971, "Vorarlberg", 700, []),  # Schruns; coords fixed (1d)
    ("Hochkönig", 47.40533, 12.90271, "Salzburg", 800, []),  # Maria Alm; coords fixed (1d)
    ("Bad Gastein", 47.17031, 13.10606, "Salzburg", 860, []),  # Bad Hofgastein; coords fixed (1d)
    ("Damüls-Mellau", 47.34998, 9.88119, "Vorarlberg", 690, []),  # Mellau; coords fixed (1d)
    ("Ski Juwel Alpbachtal Wildschönau", 47.396, 11.91, "Tyrol", 830, []),
    ("Nassfeld", 46.6129, 13.27823, "Carinthia", 600, []),  # Tröpolach; coords fixed (1d)
```

### 2c2. SWISS_SKI_AREAS (add to the existing list, around line 257)

The existing Swiss rows put the IMIS station height in the elevation slot. The new rows carry the base height until an IMIS station is chosen (section 3d), and an empty station list.

```python
    ("Grindelwald-Wengen", 46.62453, 8.03254, "Bern", 1034, []),
    ("Crans-Montana", 46.31194, 7.4825, "Valais", 1500, []),
    ("Adelboden-Lenk", 46.49306, 7.55861, "Bern", 1350, []),
    ("Gstaad", 46.47216, 7.28875, "Bern", 1050, []),
    ("Villars-Gryon-Les Diablerets", 46.29776, 7.05533, "Vaud", 1253, []),
    ("Champéry", 46.17505, 6.87137, "Valais", 1050, []),
    ("Scuol", 46.797, 10.2995, "Graubünden", 1250, []),
    ("Disentis", 46.70417, 8.85498, "Graubünden", 1130, []),
```

### 2c3. FRANCE_SKI_AREAS (add to the existing list, around line 405)

The France list has no station slot: `(name, lat, lon, region, elevation_m)`. "Pyrenees" is a new region for France.

```python
    # Northern Alps
    ("La Rosière", 45.6286, 6.84784, "Northern Alps", 1850),
    ("Valloire", 45.16528, 6.42833, "Northern Alps", 1430),
    ("Le Grand Massif (Les Carroz)", 46.025, 6.64361, "Northern Alps", 1140),
    ("Valmorel", 45.46167, 6.44333, "Northern Alps", 1400),
    ("Les Contamines-Montjoie", 45.82167, 6.72917, "Northern Alps", 1164),
    # Southern Alps
    ("Montgenèvre", 44.93083, 6.72111, "Southern Alps", 1860),
    ("Puy-Saint-Vincent", 44.83, 6.48722, "Southern Alps", 1400),
    ("Les Orres", 44.49301, 6.5571, "Southern Alps", 1650),
    ("La Grave-La Meije", 45.04611, 6.30583, "Southern Alps", 1450),
    ("Orcières-Merlette", 44.6968, 6.32186, "Southern Alps", 1850),
    # Pyrenees
    ("Saint-Lary-Soulan", 42.8125, 0.2944, "Pyrenees", 1700),
    ("Grand Tourmalet", 42.91, 0.17917, "Pyrenees", 1800),
    ("Ax 3 Domaines", 42.70192, 1.81626, "Pyrenees", 1400),
    ("Font-Romeu-Pyrénées 2000", 42.52104, 2.05967, "Pyrenees", 1777),
    # Held in 1b, not added: Peyragudes
```

### 2d. build_resorts (around line 905)

Add loops for the new lists. The current code handles only six countries. The new Swiss and French rows join existing lists, so they need no new loop; the Swiss loop (around line 908) already turns an empty station list into an empty `station_cfg`.

```python
    for name, lat, lon, region, _elev, stations in ITALY_SKI_AREAS:
        r = base(name, lat, lon, region, "Italy")
        r["station_cfg"] = []
        out.append(r)
    for country, areas in (("Germany", GERMANY_SKI_AREAS), ("Slovenia", SLOVENIA_SKI_AREAS)):
        for name, lat, lon, region, _elev, stations in areas:
            r = base(name, lat, lon, region, country)
            r["station_cfg"] = []
            out.append(r)
```

Also needed, in the same file:

- `scripts/validate_forecasts_v2.py`, line 57: `COUNTRIES` must include "Italy", "Germany" and "Slovenia". Without this, the validator rejects every new row.
- `scripts/fetch_data_v2.py`, around line 2895: the country list for the depth summary (`base_depth`) must include the three new countries.
- `scripts/fetch_data_v2.py`, around line 2914: add source entries for the new providers, once any are wired.
- `scripts/fetch_data_v2.py`, around line 2707: the depth chain falls through to the "France, Spain" branch for any unknown country. That is fine at launch. Once a provider is wired, give it its own branch.
- `scripts/fetch_data_v2.py`, around line 3014: `alpine = {"Austria", "Switzerland"}` decides which resorts get the ICON-D2 model. ICON-D2 covers Germany and parts of northern Italy. Decide whether to add Germany and Italy there. This is a choice, not a requirement.

### 2e. RESORT_ELEVATIONS (around line 456)

Format: a source comment above each entry, then `"Name": (base_m, summit_m, "note"),`. Base and summit are whole metres.

```python
    # USA (new)
    # https://news.vailresorts.com/stowe-mountain-resort-tip-sheet
    "Stowe": (475, 1340, "Vail Resorts tip sheet (base 475 m, summit 1,340 m)"),
    # https://en.wikipedia.org/wiki/Sugarbush_Resort
    "Sugarbush": (452, 1244, "en.wikipedia.org (base 452 m, summit 1,244 m)"),
    # https://www.stratton.com (mountain statistics); en.wikipedia.org/wiki/Stratton_Mountain_Resort
    # Coords: OSM Main Base Lodge. USGS 10 m model gives about 647 m at the lodge; official base kept.
    "Stratton": (571, 1181, "stratton.com official (base 571 m, summit 1,181 m); skiresort.com gives 600 m base"),
    # https://en.wikipedia.org/wiki/Okemo_Mountain ; skiresort.com 348-1,019 m
    # Coords: OSM main base. 346 m is the lowest lift at Jackson Gore, 2 km NNE.
    "Okemo": (346, 1019, "en.wikipedia.org (base 346 m, summit 1,019 m); skiresort.com agrees within 2 m"),
    # https://en.wikipedia.org/wiki/Mount_Snow ; skiresort.com 585-1,095 m
    # Coords: OSM Main Base Lodge (the Wikipedia infobox point is the summit, not used).
    "Mount Snow": (579, 1097, "en.wikipedia.org (base 579 m, summit 1,097 m); skiresort.com agrees within 6 m"),
    # https://en.wikipedia.org/wiki/Jay_Peak_Resort ; skiresort.com 563-1,175 m
    # Coords: OSM Aerial Tram bottom station. USGS 10 m model gives 565 m there.
    "Jay Peak": (553, 1177, "en.wikipedia.org (base 553 m, summit 1,177 m)"),
    # https://en.wikipedia.org/wiki/Gore_Mountain_(ski_resort) ; skiresort.com 304-1,097 m
    # Coords: OSM Base Lodge (about 465 m). 304 m is the North Creek Ski Bowl lift, 2.5 km east.
    "Gore Mountain": (304, 1097, "en.wikipedia.org and skiresort.com (base 304 m, summit 1,097 m)"),
    # https://en.wikipedia.org/wiki/Whiteface_Mountain ; skiresort.com 371-1,340 m
    # Coords: OSM base lodge and Cloudsplitter Gondola bottom. USGS 10 m model gives 372 m there.
    "Whiteface": (372, 1337, "en.wikipedia.org (base 372 m, lift-served top 1,337 m)"),
    # https://en.wikipedia.org/wiki/Loon_Mountain_Ski_Resort
    # Coords fixed (plan 1d): OSM Village of Loon Mountain. USGS 10 m model gives 296 m there.
    "Loon Mountain": (260, 930, "en.wikipedia.org (base 260 m, summit 930 m)"),
    # https://en.wikipedia.org/wiki/Sunday_River_(ski_resort)
    "Sunday River": (244, 957, "en.wikipedia.org (base 244 m, summit 957 m)"),
    # https://en.wikipedia.org/wiki/Sugarloaf_(ski_resort)
    # Coords fixed (plan 1d): OSM Sugarloaf Base Lodge (about 536 m); 432 m is the lowest lift.
    "Sugarloaf": (432, 1291, "en.wikipedia.org (base 432 m, summit 1,291 m)"),
    # https://en.wikipedia.org/wiki/Steamboat_Ski_Resort
    "Steamboat": (2103, 3221, "en.wikipedia.org (base 2,103 m, summit 3,221 m)"),
    # https://en.wikipedia.org/wiki/Winter_Park_Resort
    "Winter Park": (2743, 3676, "en.wikipedia.org (base 2,743 m, summit 3,676 m)"),
    # https://en.wikipedia.org/wiki/Copper_Mountain_(Colorado)
    "Copper Mountain": (2960, 3792, "en.wikipedia.org (base 2,960 m, summit 3,792 m)"),
    # https://en.wikipedia.org/wiki/Keystone_Resort
    "Keystone": (2829, 3782, "en.wikipedia.org (base 2,829 m, summit 3,782 m)"),
    # https://en.wikipedia.org/wiki/Arapahoe_Basin
    "Arapahoe Basin": (3286, 3978, "en.wikipedia.org (base 3,286 m, summit 3,978 m)"),
    # https://en.wikipedia.org/wiki/Crested_Butte_Mountain_Resort
    "Crested Butte": (2858, 3707, "en.wikipedia.org (base 2,858 m, summit 3,707 m)"),
    # https://en.wikipedia.org/wiki/Snowbasin
    "Snowbasin": (1948, 2850, "en.wikipedia.org (base 1,948 m, summit 2,850 m)"),
    # https://en.wikipedia.org/wiki/Solitude_Mountain_Resort
    "Solitude": (2435, 3197, "en.wikipedia.org (base 2,435 m, summit 3,197 m)"),
    # https://en.wikipedia.org/wiki/Brighton_Ski_Resort
    "Brighton": (2669, 3277, "en.wikipedia.org (base 2,669 m, summit 3,277 m)"),
    # https://en.wikipedia.org/wiki/Powder_Mountain
    # Upside-down resort: coords are the top village (about 2,500 m); 2,103 m is the lowest lift.
    "Powder Mountain": (2103, 2713, "en.wikipedia.org (base 2,103 m, lift-served top 2,713 m)"),
    # https://en.wikipedia.org/wiki/Taos_Ski_Valley,_New_Mexico
    # Coords fixed (plan 1d): OSM Taos Ski Valley admin point, USGS 10 m model 2,844 m.
    "Taos Ski Valley": (2804, 3804, "en.wikipedia.org (base 2,804 m, summit 3,804 m)"),
    # https://en.wikipedia.org/wiki/Alyeska_Resort
    "Alyeska": (76, 1201, "en.wikipedia.org (base 76 m, summit 1,201 m)"),
    # https://en.wikipedia.org/wiki/Schweitzer_Mountain
    # Base fixed (plan 1d): 1,207 m is the Stella chair; coords are the main village (1,433 m).
    "Schweitzer": (1207, 1951, "en.wikipedia.org (base 1,207 m, summit 1,951 m)"),
    # Held in plan 1b, no entry yet: Deer Valley, Killington, Northstar California.
    # Italy (new)
    # https://en.wikipedia.org/wiki/Cortina_d%27Ampezzo
    "Cortina d'Ampezzo": (1224, 2930, "en.wikipedia.org (base 1,224 m, summit 2,930 m)"),
    # https://en.wikipedia.org/wiki/Selva_di_Val_Gardena
    "Val Gardena": (1563, 2518, "en.wikipedia.org (village base 1,563 m); skiresort.com (summit 2,518 m)"),
    # https://en.wikipedia.org/wiki/Corvara_in_Badia
    "Alta Badia": (1568, 2778, "en.wikipedia.org (village base 1,568 m); skiresort.com (summit 2,778 m)"),
    # https://en.wikipedia.org/wiki/Canazei
    "Val di Fassa": (1450, 2485, "en.wikipedia.org (base 1,450 m); snowtrex.co.uk (summit 2,485 m)"),
    # https://en.wikipedia.org/wiki/Arabba
    # Coords and base fixed (plan 1d): OSM Arabba village, EU-DEM 1,603 m (Wikipedia gave 1,645 m).
    "Arabba-Marmolada": (1602, 3269, "EU-DEM at Arabba village (base 1,602 m); skiresort.com (summit 3,269 m)"),
    # https://en.wikipedia.org/wiki/San_Martino_di_Castrozza
    "San Martino di Castrozza": (1487, 2357, "en.wikipedia.org (base 1,487 m); snowtrex.co.uk (summit 2,357 m)"),
    # https://en.wikipedia.org/wiki/Sestriere
    # Coords fixed (plan 1d): OSM Sestriere village, EU-DEM 2,039 m.
    "Sestriere": (2035, 2800, "en.wikipedia.org (base 2,035 m); Via Lattea summit about 2,800 m"),
    # https://en.wikipedia.org/wiki/Breuil-Cervinia
    "Breuil-Cervinia": (2050, 3480, "en.wikipedia.org (base 2,050 m, Testa Grigia 3,480 m)"),
    # https://en.wikipedia.org/wiki/Mont_Blanc_de_Courmayeur
    "Courmayeur": (1224, 3466, "en.wikipedia.org (base 1,224 m); Skyway top 3,466 m (pistes top about 2,755 m)"),
    # https://en.wikipedia.org/wiki/Champoluc
    # Coords fixed (plan 1d): OSM Champoluc, EU-DEM 1,561 m.
    "Monterosa Ski": (1560, 3275, "en.wikipedia.org (Champoluc base 1,560 m); Punta Indren 3,275 m"),
    # https://en.wikipedia.org/wiki/Livigno
    "Livigno": (1816, 2798, "en.wikipedia.org (base 1,816 m); skiresort.com (summit 2,798 m)"),
    # https://en.wikipedia.org/wiki/Bormio
    "Bormio": (1225, 3017, "en.wikipedia.org (base 1,225 m); Cima Bianca 3,017 m (sources give 3,012 m)"),
    # https://en.wikipedia.org/wiki/Madonna_di_Campiglio
    "Madonna di Campiglio": (1522, 2504, "en.wikipedia.org (base 1,522 m); trentino.com (summit 2,504 m)"),
    # https://en.wikipedia.org/wiki/Passo_del_Tonale
    "Passo del Tonale": (1883, 3000, "en.wikipedia.org (base 1,883 m); Presena Glacier about 3,000 m"),
    # https://en.wikipedia.org/wiki/Solda
    "Solda-Sulden": (1906, 3250, "en.wikipedia.org (base 1,906 m); skiresort.com (ski area top 3,250 m)"),
    # https://en.wikipedia.org/wiki/Val_Senales
    # Coords fixed (plan 1d): OSM Kurzras-Maso Corto, EU-DEM 2,011 m.
    "Schnalstal": (2011, 3212, "bergfex.com (base 2,011 m, summit 3,212 m)"),
    # https://en.wikivoyage.org/wiki/Seiser_Alm ; skiresort.com Seiser Alm (Compatsch 1,860 m)
    # Base and coords fixed (plan 1d): OSM Compatsch, EU-DEM 1,841 m. 1,060 m was the Seis valley gondola.
    "Alpe di Siusi": (1850, 2245, "skiresort.com (Compatsch 1,860 m) and EU-DEM (1,841 m); en.wikivoyage.org (summit 2,245 m; skiresort says 2,220 m)"),
    # https://www.skiresort.com (Kronplatz 973-2,275 m); en.wikipedia.org/wiki/Kronplatz
    # Coords: OSM Reischach-Riscone, EU-DEM 954 m. No source states 973 m is the Reischach station.
    "Kronplatz": (973, 2275, "skiresort.com (base 973 m, summit 2,275 m); Wikipedia confirms summit"),
    # Held in plan 1b, no entry yet: La Thuile.
    # Austria (new)
    # https://en.wikipedia.org/wiki/Saalbach-Hinterglemm
    # Base fixed (plan 1d): EU-DEM 995 m at the village. 830 m is the Skicircus low point (Leogang, Fieberbrunn).
    "Saalbach-Hinterglemm": (1003, 2096, "en.wikipedia.org (village base 1,003 m, summit 2,096 m)"),
    # https://en.wikipedia.org/wiki/Mayrhofen
    "Mayrhofen": (630, 2500, "en.wikipedia.org (base 630 m); skiresort.com (summit 2,500 m)"),
    # https://en.wikipedia.org/wiki/Zillertal_Arena
    # Coords fixed (plan 1d): OSM Zell am Ziller, EU-DEM 578 m.
    "Zillertal Arena": (580, 2500, "en.wikipedia.org (base 580 m); skiresort.com (summit 2,500 m)"),
    # https://en.wikipedia.org/wiki/Hintertux_Glacier
    # Coords fixed (plan 1d): OSM Hintertux village, EU-DEM 1,496 m.
    "Hintertux Glacier": (1500, 3250, "en.wikipedia.org (base 1,500 m, summit 3,250 m)"),
    # https://www.stubaital.at (Talstation Stubaier Gletscher 1,749 m); bergfex (Gamsgarten 1,750 m)
    # Coords: OSM Mutterberg, EU-DEM 1,739 m. Not the OSM 'Stubaier Gletscher' point (on the glacier).
    "Stubai Glacier": (1750, 3210, "stubaital.at official (base 1,749 m); skiresort.com (summit 3,210 m, Top of Tyrol)"),
    # https://en.wikipedia.org/wiki/Kaprun
    "Kitzsteinhorn": (768, 3029, "en.wikipedia.org (Kaprun 768 m base per plan; summit 3,029 m)"),
    # https://en.wikipedia.org/wiki/Serfaus
    # Base fixed (plan 1d): EU-DEM 1,434 m at the coords. 1,200 m is the Ladis valley station.
    "Serfaus-Fiss-Ladis": (1429, 2828, "en.wikipedia.org (village base 1,429 m, summit 2,828 m)"),
    # https://en.wikipedia.org/wiki/SkiWelt
    "SkiWelt Wilder Kaiser-Brixental": (620, 1869, "en.wikipedia.org (base 620 m, summit 1,869 m)"),
    # https://en.wikipedia.org/wiki/Silvretta_Montafon
    # Coords fixed (plan 1d): OSM Schruns, EU-DEM 692 m.
    "Silvretta Montafon": (700, 2430, "en.wikipedia.org (base 700 m, summit 2,430 m)"),
    # https://en.wikipedia.org/wiki/Hochk%C3%B6nig_(ski_area)
    # Coords fixed (plan 1d): OSM Maria Alm, EU-DEM 799 m.
    "Hochkönig": (800, 1900, "en.wikipedia.org (base 800 m); onthesnow.co.uk (summit 1,900 m; Wikipedia says 2,000 m)"),
    # https://en.wikipedia.org/wiki/Bad_Hofgastein
    # Coords fixed (plan 1d): OSM Bad Hofgastein, EU-DEM 856 m.
    "Bad Gastein": (860, 2300, "en.wikipedia.org (base 860 m, summit 2,300 m)"),
    # https://www.skiresort.info/ski-resort/damuels-mellau/
    # Coords fixed (plan 1d): OSM Mellau village, EU-DEM 689 m (old point was Damüls, 1,413 m).
    "Damüls-Mellau": (690, 2050, "skiresort.info (base 690 m, summit 2,050 m)"),
    # https://www.skiresort.info/ski-resort/ski-juwel-alpbachtal-wildschoenau/test-result/size/
    "Ski Juwel Alpbachtal Wildschönau": (830, 2030, "skiresort.info (base 830 m, summit 2,030 m)"),
    # https://en.wikipedia.org/wiki/Nassfeld ; skiresort.com Nassfeld-Hermagor 600-2,002 m
    # Coords fixed (plan 1d): OSM Tröpolach, EU-DEM 609 m (Millennium-Express valley station).
    "Nassfeld": (600, 2002, "skiresort.com (base 600 m at Tröpolach, summit 2,002 m)"),
    # Switzerland (new)
    # https://en.wikipedia.org/wiki/Lauberhorn ; snow-online.com 1,034-2,472 m
    # Coords: OSM Grindelwald station (Wikidata Q64004 129 m away). skiresort.com 944-2,400 m; bergfex 937-2,500 m.
    "Grindelwald-Wengen": (1034, 2472, "Wikidata and OSM (Grindelwald 1,034 m); Lauberhorn 2,472 m"),
    # https://www.crans-montana.ch (1,500-3,000 m); skiresort.com 1,510-2,927 m
    # Coords: Wikidata Q688557 (Montana side). OSM point 1.2 km away on the Crans side.
    "Crans-Montana": (1500, 2927, "crans-montana.ch official (base 1,500 m); Plaine Morte 2,927 m"),
    # https://www.skiresort.com/en/ski-resort/adelboden-lenk-chuenisbaergli-silleren-hahnenmoos-metsch
    # Coords: Wikidata Q68994 and OSM, 39 m apart. Whole lift pass tops out at 2,362 m.
    "Adelboden-Lenk": (1350, 2200, "Wikidata and OSM (Adelboden 1,350 m); skiresort.com (summit 2,200 m)"),
    # skiresort.info lift l92139 (La Videmanette 2,151 m); maps.adac.de
    # Coords: OSM Gstaad village only. Excludes Glacier 3000.
    "Gstaad": (1050, 2151, "Wikidata, OSM and SnowTrex (base 1,050 m); La Videmanette 2,151 m"),
    # https://www.skiresort.com/en/ski-resort/villars-gryon-les-diablerets (1,224-2,120 m)
    # Coords: OSM Villars-sur-Ollon (station 220 m away). Excludes Glacier 3000.
    "Villars-Gryon-Les Diablerets": (1253, 2120, "Wikidata (Villars 1,253 m); Grand Chamossaire 2,120 m"),
    # https://www.portesdusoleil.com (Champéry 1,050 m); en.wikipedia.org/wiki/Pointe_des_Mossettes
    # Coords: Wikidata Q70099 (OSM 212 m away). Whole Portes du Soleil tops at 2,466 m (Avoriaz).
    "Champéry": (1050, 2277, "portesdusoleil.com official (base 1,050 m); Pointe des Mossettes 2,277 m"),
    # https://www.skiresort.com/en/ski-resort/scuol-motta-naluns ; bergfex 2,785 m
    # Coords: Wikidata Q69079 and OSM, 72 m apart.
    "Scuol": (1250, 2785, "skiresort.com and bergfex (base 1,250 m, summit 2,785 m)"),
    # https://www.skiresort.com/en/ski-resort/disentis (1,227-2,833 m); snowtrex 2,833 m (Péz Ault)
    # Coords: OSM Disentis/Mustér village (Wikidata 637 m away). Lowest lift at 1,227 m.
    "Disentis": (1130, 2833, "Wikidata and OSM (village 1,130 m); skiresort.com (summit 2,833 m)"),
    # France (new)
    # https://www.larosiere.ski (Mont Valaisan 2,800 m); skiresort.com 1,190-2,800 m
    # Coords: OSM La Rosière hamlet (Wikidata 190 m away).
    "La Rosière": (1850, 2800, "larosiere.ski official (summit 2,800 m); Wikipedia (village 1,850 m)"),
    # https://www.valloire.net ; skiresort.info galibier-thabor-valmeinier-valloire 1,430-2,750 m
    # Coords: Wikidata Q819204 (OSM 107 m away). Official FAQ gives the village at 1,450 m.
    "Valloire": (1430, 2750, "valloire.net official (1,430-2,750 m)"),
    # https://www.skiresort.com/en/ski-resort/le-grand-massif-flaine-les-carroz-morillon-samoens-sixt
    # Coords: Wikidata Q3079691 Les Carroz (OSM 407 m away). Summit is Grandes Platières.
    "Le Grand Massif (Les Carroz)": (1140, 2480, "Grand Massif Domaines Skiables via alti-mag (Les Carroz 1,140 m); skiresort.com (summit 2,480 m)"),
    # https://www.valmorel.com (Grand Domaine 1,250-2,550 m); skiresort.info 1,250-2,514 m
    # Coords: Wikidata Q2509456 only. Village 1,400 m (alti-mag, j2ski, snowtrex); Wikidata says 1,350 m.
    "Valmorel": (1400, 2550, "valmorel.com official (summit 2,550 m); village 1,400 m"),
    # https://www.skiresort.com/en/ski-resort/les-contamines-hauteluce ; bergfex 2,487 m
    # Coords: Wikidata Q477987 (OSM 144 m away). Summit sources range 2,437-2,500 m.
    "Les Contamines-Montjoie": (1164, 2487, "France Montagnes official (base 1,164 m); bergfex (summit 2,487 m)"),
    # https://www.montgenevre.com (village 1,860 m; ski area to 2,648 m; Via Lattea 2,823 m)
    # Coords: Wikidata Q933254 and OSM, 66 m apart.
    "Montgenèvre": (1860, 2648, "montgenevre.com official (base 1,860 m, summit 2,648 m)"),
    # https://www.puysaintvincent.com (1,400-2,750 m); skiresort.com 1,407-2,677 m
    # Coords: Wikidata Q1086218 (OSM 484 m away). Base is the 1400 village.
    "Puy-Saint-Vincent": (1400, 2750, "puysaintvincent.com official (1,400-2,750 m)"),
    # https://www.skiresort.com/en/ski-resort/les-orres (1,550-2,720 m); ski.fr and esi-ski agree
    # Coords: OSM 'Centre Station 1650'. Not the Wikidata commune point (old village, 2.3 km away).
    "Les Orres": (1650, 2720, "skiresort.com (1650 station, summit 2,720 m)"),
    # https://www.hautesvallees.com (cable car 1,450-3,200 m); skiresort.com 1,470-3,534 m
    # Coords: Wikidata Q624323 (OSM 101 m away). Glacier T-bar to about 3,550 m not always running.
    "La Grave-La Meije": (1450, 3200, "hautesvallees.com official (cable car 1,450-3,200 m)"),
    # https://www.orcieres.com (1,850-2,725 m); skiresort.com/en/ski-resort/orcieres-merlette-1850
    # Coords: Wikidata Q3355386 and OSM, about 35 m apart. Not the Orcières valley village.
    "Orcières-Merlette": (1850, 2725, "orcieres.com official and skiresort.com (1,850-2,725 m)"),
    # https://www.saintlary.com (1,700-2,515 m); skiresort.com 1,597-2,515 m
    # Coords: Wikidata Q3462692 and OSM Pla d'Adet, 26 m apart. Not the valley village (830 m).
    "Saint-Lary-Soulan": (1700, 2515, "saintlary.com official (Pla d'Adet 1,700 m, summit 2,515 m)"),
    # https://www.n-py.com (2,500 m); skiresort.com 1,250-2,500 m; en.wikipedia.org/wiki/La_Mongie
    # Coords: Wikidata Q356665 La Mongie (OSM 245 m away). Barèges side is about 1,250 m.
    "Grand Tourmalet": (1800, 2500, "Wikipedia and alti-mag (La Mongie 1,800 m); n-py.com official (summit 2,500 m)"),
    # https://en.wikipedia.org/wiki/Ax_3_Domaines ; ariegepyrenees.com; bergfex 2,400 m; skiresort.com 1,380-2,346 m
    # Coords: Wikidata Q791836 (OSM 172 m away). Bonascre plateau, not Ax-les-Thermes town.
    "Ax 3 Domaines": (1400, 2400, "Wikipedia and Ariège tourism (Bonascre 1,400 m, summit 2,400 m)"),
    # https://www.altiservice.com (Pyrénées 2000 base 1,777 m; Roc de la Calme 2,213 m); skiresort.info 1,717-2,213 m
    # Coords: OSM Pyrénées 2000 hamlet only (Wikidata rounded, 911 m away).
    "Font-Romeu-Pyrénées 2000": (1777, 2213, "altiservice.com official (base 1,777 m, summit 2,213 m)"),
    # Held in plan 1b, no entry yet: Peyragudes.
    # Germany (new)
    # https://en.wikipedia.org/wiki/Oberstdorf
    "Oberstdorf-Kleinwalsertal": (820, 2224, "en.wikipedia.org (base 820 m); Nebelhorn 2,224 m"),
    # https://en.wikipedia.org/wiki/Garmisch_Classic
    "Garmisch Classic": (700, 2050, "en.wikipedia.org (base 700 m, summit 2,050 m)"),
    # https://de.wikipedia.org/wiki/Skigebiet_Zugspitze ; skiresort.com 2,000-2,650 m
    # Coords: OSM Zugspitzplatt hub (about 2,576 m). 2,000 m is the Brunntal lift, 1.5 km east.
    # de.wikipedia's 2,720 m top is out of date (Schneefernerkopf lifts closed after 2023/24).
    "Zugspitze": (2000, 2650, "de.wikipedia.org and skiresort.com (base 2,000 m); skiresort.com (summit 2,650 m)"),
    # https://www.feldberg-erlebnis.de (14 lifts, area to 1,450 m); skiresort.com 950-1,450 m
    # Coords: OSM Talstation Feldbergbahn-Seebuck (about 1,280 m). 950 m is at Fahl, 47.8561,8.0049.
    "Feldberg": (950, 1450, "skiresort.com (base 950 m); feldberg-erlebnis.de official (summit 1,450 m)"),
    # https://en.wikipedia.org/wiki/Winterberg
    "Winterberg": (630, 820, "skiresort.com (base 630 m, summit 820 m)"),
    # Slovenia (new)
    # https://en.wikipedia.org/wiki/Kranjska_Gora_Ski_Resort
    "Kranjska Gora": (807, 1282, "en.wikipedia.org (base 811 m, top 1,295 m; scout base 807 m, top 1,282 m)"),
```

The Kranjska Gora note is worth a second look. The scout's numbers differ from Wikipedia's in the note, so check which pair is used before committing.

### 2f. ONTHESNOW_URLS (around line 627)

Austria and Switzerland have no entries today, and Italy, Germany and Slovenia have no known slug pattern. Leave them out at launch. For the new US resorts, the scouts' OnTheSnow URLs can be turned into candidates by replacing `/ski-resort` with `/skireport`. These are not verified. A fetch has to confirm each one first.

```python
    "Stowe": _OTS + "vermont/stowe-mountain-resort/skireport",           # unverified
    "Sugarbush": _OTS + "vermont/sugarbush/skireport",                   # unverified
    "Sunday River": _OTS + "maine/sunday-river/skireport",               # unverified
    "Sugarloaf": _OTS + "maine/sugarloaf/skireport",                     # unverified
    "Winter Park": _OTS + "colorado/winter-park-resort/skireport",       # unverified
    "Crested Butte": _OTS + "colorado/crested-butte-mountain-resort/skireport",  # unverified
    "Keystone": _OTS + "colorado/keystone/skireport",                    # unverified
    "Arapahoe Basin": _OTS + "colorado/arapahoe-basin-ski-area/skireport",       # unverified
    "Snowbasin": _OTS + "utah/snowbasin/skireport",                      # unverified
    "Solitude": _OTS + "utah/solitude-mountain-resort/skireport",        # unverified
    "Brighton": _OTS + "utah/brighton-resort/skireport",                 # unverified
    "Powder Mountain": _OTS + "utah/powder-mountain/skireport",          # unverified
    "Alyeska": _OTS + "alaska/alyeska-resort/skireport",                 # unverified
    "Schweitzer": _OTS + "idaho/schweitzer/skireport",                   # unverified
    # Held in 1b, add once verified: Killington, Deer Valley, Northstar California.
    # Stratton, Okemo, Mount Snow, Jay Peak, Gore Mountain, Whiteface: no scout URL, slug to find.
```

## 3. Data wiring per country

Here is what covers each new resort today, and what is missing.

**a. USA West (Colorado, Utah, New Mexico, Alaska, Idaho; California once Northstar is verified).** SNOTEL covers these, through the AWDB API (the same path the fetcher uses today). Each resort needs an AWDB station lookup: find SNOTEL sites within 15 km, check their elevation against the rule (at most 300 m below base or 300 m above summit), and record the triplet and weight. Alyeska probably has no SNOTEL site; check the AWDB list. Schweitzer may also have none. Until the lookup is done, use the empty list and show "no measured depth".

**b. USA East (Vermont, New York, New Hampshire, Maine).** No station source is wired. This now covers 11 resorts, including Gore Mountain, Whiteface, Stratton, Okemo, Mount Snow and Jay Peak from the verification pass. Options, all unverified:

- NOHRSC gridded snow analysis from the US National Weather Service. Covers the whole country, daily, and can be read at a point. Best candidate for a measured depth.
- NOAA GHCN-Daily snow depth from COOP stations. Free, but reports arrive late and stations are sparse at ski elevations.
- New York State Mesonet, for Whiteface and Gore Mountain. Check whether a station sits within 15 km.
- Vermont and New Hampshire state networks, if any public feed exists. Not checked.

At launch, East resorts show "no measured depth", and the resort report is used when there is one.

**c. Austria.** GeoSphere TAWES covers Salzburg, Carinthia and Vorarlberg. Land Tirol covers Tyrol. Each new resort needs a station lookup:

- Tyrol (Mayrhofen, Zillertal Arena, Hintertux, Stubai, Serfaus, SkiWelt, Ski Juwel): Land Tirol stations.
- Salzburg (Saalbach, Kitzsteinhorn, Hochkönig, Bad Gastein): GeoSphere stations.
- Vorarlberg (Silvretta Montafon, Damüls-Mellau): GeoSphere stations.
- Carinthia (Nassfeld): GeoSphere stations.

Several Austrian coordinates moved in the verification pass (Zillertal Arena, Hintertux, Silvretta Montafon, Hochkönig, Bad Gastein, Damüls-Mellau, Nassfeld), so run the 15 km station search from the new points.

The fetcher already accepts both providers, so only the IDs are missing.

**d. Switzerland.** SLF IMIS covers the country. The eight new rows (Grindelwald-Wengen through Disentis) now have coordinates, so each one needs an IMIS station lookup before its run. Until then the station list is empty.

**d2. France.** The 14 new French rows join FRANCE_SKI_AREAS and use the existing France depth chain. They need Météo-France station IDs, looked up from the new coordinates, before their run. The Pyrenees rows are the first French resorts outside the Alps.

**e. Italy, Germany, Slovenia.** Nothing is wired. Candidate open sources to research. None is verified.

- Italy: South Tyrol open data (Südtirol, for Val Gardena, Alta Badia, Solda, Schnalstal, Alpe di Siusi and Kronplatz), Veneto environmental agency open data (ARPAV, for the Cortina and Arabba areas), Lombardy environmental agency open data (ARPA Lombardia, for Bormio and Livigno), Piedmont (ARPA Piemonte, for Sestriere), and Aosta Valley regional open data (for Breuil-Cervinia, Courmayeur and Monterosa; La Thuile once verified). Trentino (for Madonna di Campiglio, Passo del Tonale, Val di Fassa and San Martino) is also worth a look.
- Germany: DWD Climate Data Center for daily snow depth (the federal open data source). Bavarian state networks for Oberstdorf, Garmisch and Zugspitze. The Baden-Württemberg state network for Feldberg. The North Rhine-Westphalia state network for Winterberg.
- Slovenia: ARSO open data for Kranjska Gora.

**f. What stays honestly empty at launch.** Every new Italy, Germany and Slovenia resort, and every East US resort, shows no measured depth. The build should add a `DEPTH_COVERAGE_NOTES` entry for each, the same way existing resorts with no station get one. Without it, the output says nothing about why depth is missing.

**g. Timezone.** Nothing to wire. Open-Meteo returns the zone.

**h. OnTheSnow.** The US slug pattern is `onthesnow.com/<state>/<resort-slug>/skireport`. The Alps pattern (onthesnow.co.uk, with a region group such as `northern-alps` or `aragon`) exists for France and Spain only. Austria and Switzerland have no entries at all. For Italy, Germany and Slovenia, the slug pattern is unknown, so each one needs a fetch check. Leave these empty until checked.

## 4. Scale math

**Current run.** 86 resorts. 258 elevation points for the daily call and the same for the hourly call (each resort has base, mid and summit, so 3 points). The fetcher sends 10 resorts per call, so 30 points per call. The summit-only "models" call uses 86 points. The last run made 31 Open-Meteo requests: 9 daily, 9 hourly, 10 models, and 3 more from verification and retries. It took 685 seconds, or about 11.4 minutes.

**After (170 resorts).** Points per call stay at 30 (10 resorts). Calls go up:

- Daily: 17 calls (170 resorts in batches of 10).
- Hourly: 17 calls.
- Models: 43 Austria and Switzerland resorts (alpine group: 21 today plus 14 new Austria and 8 new Switzerland) make 5 calls. The other 127 make 13 calls. Total 18.
- Overhead from verification and retries: about 3.

That is about 55 requests per run, against 31 today. Per day at four runs: about 220, against 124.

Points per run: 1,190 (510 + 510 + 170), against 602 today.

**Open-Meteo daily limit.** The free non-commercial plan allows 10,000 API calls per day. Open-Meteo's docs decide how a multi-coordinate request counts. If each request counts once, the new total is about 220 a day, well under the limit. If each coordinate counts, it is about 4,760 a day (1,190 points times 4 runs), still under the limit but about 48% of it. Check the current pricing page before relying on either number.

**Runtime.** Straight-line from 685 seconds gives about 1,354 seconds (22.6 minutes) for 170 resorts. The workflow timeout is 25 minutes. That leaves under 3 minutes, which one slow provider retry would use up. The estimate is rough. Raising the timeout is now required, not optional (see below).

**Recommendations.**

- Keep `OM_BATCH_RESORTS = 10`. It keeps each request small and a failed call contained. Raising it to 20 would halve the call count, but a bad batch then loses 20 resorts instead of 10.
- Raise `timeout-minutes` from 25 to 35 in the same change that adds Italy. That gives room for the run-time growth without touching anything else. At 170 resorts the estimate is 23 minutes, so 35 leaves about 12 minutes of slack.
- Keep the 4x daily schedule unchanged.

Combined numbers with the Asia plan (220 resorts after the 27 Nov 2026 ruling that restored the Asia plan's 8 size-dropped rows, Open-Meteo volume against the free tier, and run time) are in `docs/resort_expansion_asia.md`, section 4. That ruling changes nothing in this plan: no Alps or US row was held back for size — the five held rows in 1b are all coordinate or base problems, and the size cuts here happened at scouting, before any data was verified.

## 5. Rollout

Each phase is one fetcher run that verifies the phase. A phase is done when its resorts show a forecast, a summit, and a depth or an honest "no measured depth".

**Phase 1: Italy (18 resorts).** The biggest gap. Work: add ITALY_SKI_AREAS, RESORT_ELEVATIONS, the build_resorts loop, the validator country entry, and the timeout raise (section 4). Forecasts come from Open-Meteo, so this run verifies forecast coverage only. Depth stays empty until a source is wired (section 3e). Two Open-Meteo batches per kind. Includes Kronplatz; La Thuile waits in 1b.

**Phase 2: US East (11 resorts).** Stowe, Sugarbush, Stratton, Okemo, Mount Snow, Jay Peak, Gore Mountain, Whiteface, Loon, Sunday River and Sugarloaf. Depth stays empty until a source is wired (section 3b). Two Open-Meteo batches per kind. Killington waits in 1b. The remaining scout gap (Bromley, Mad River Glen, Cannon) needs a fresh scout pass.

**Phase 3: US West (13 resorts).** Colorado (6), Utah (4), Taos, Alyeska, Schweitzer. The AWDB station lookups (section 3a) come before the run. Without them, these resorts show no depth. Deer Valley and Northstar wait in 1b.

**Phase 4: Austria (14 resorts).** Add the GeoSphere and Land Tirol station IDs (section 3c) before the run, using the corrected coordinates. The base questions for Saalbach, Serfaus and Nassfeld are settled (section 1d). Includes Stubai Glacier.

**Phase 5: Germany (5) and Slovenia (1).** Forecasts first, depth after a source is wired. Add the country names to the validator and the depth summary. Includes Zugspitze and Feldberg.

**Phase 6: Switzerland (8) and France (14).** No longer blocked on coordinates. Add the SLF IMIS IDs (Switzerland) and the Météo-France IDs (France) before the run. Peyragudes waits in 1b.

**Later: the 5 held rows.** Peyragudes, Deer Valley, Killington, Northstar California and La Thuile join whichever phase matches their country once a base coordinate is sourced (reasons in 1b).

Order by value: Italy first, US East second, then the gap-fills (US West, Austria, then Germany and Slovenia). Switzerland and France come last because they need station lookups first, and both countries already have strong coverage.

Check: 18 + 11 + 13 + 14 + 6 + 22 = 84.
