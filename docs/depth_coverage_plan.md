# Snow depth coverage plan for `scripts/fetch_data_v2.py`

Written 27 Nov 2026. This is a plan only; the fetcher was not edited.

## Result

Today 41 of 86 resorts have a base depth. After this plan, 71 will have one on a run like today's, and 74 once three French resort stations restart for the season in mid December. Resorts with a measured recent-snow figure go from 43 to 72.

Every source adopted here was called for real on 27 Nov 2026 and returned a plausible value, needs no API key, and allows reuse with credit. Every station distance below was recomputed with a haversine against the resort coordinates in `data/forecasts_v2.json`. Nothing is estimated: where no station qualifies, the resort stays empty and says why.

| Country | Resorts | Depth now | Depth after (today) | Depth after season start | Recent measured now | Recent measured after |
|---|---|---|---|---|---|---|
| USA | 22 | 21 | 21 (+Telluride, −Aspen) | 21 | 21 | 22 |
| Austria | 10 | 5 | 9 (+5, −Obertauern) | 9 | 5 | 9 |
| Switzerland | 11 | 10 | 11 | 11 | 10 | 11 |
| Japan | 10 | 3 | 9 | 9 | 5 | 9 |
| France | 20 | 2 | 17 | 20 | 2 | 17 (20) |
| Spain | 13 | 0 | 4 | 4 | 0 | 4 |
| **Total** | **86** | **41** | **71** | **74** | **43** | **72 (75)** |

Two resorts lose a depth they show today, on purpose: Aspen (its only station is 32.6 km away) and Obertauern (49.7 km away). Those numbers were never about the resort.

---

## 0. Rules that apply to every provider

### 0.1 Station acceptance

Replace `MAX_STATION_DISTANCE_KM = 60.0` with `15.0`, and add an elevation check in `_accept_station`:

```python
MAX_STATION_DISTANCE_KM = 15.0
STATION_ELEV_BELOW_BASE_M = 300   # station may sit at most 300 m below the resort base
STATION_ELEV_ABOVE_TOP_M = 300    # and at most 300 m above the summit (base + 1500 if summit unknown)
```

A station outside either limit is dropped with a note in `resort["notes"]` (the existing pattern). Applying these two rules to the stations configured today removes:

| Resort | Station | Why |
|---|---|---|
| Obertauern | GeoSphere 11222 Flattnitz | 49.7 km. Currently used as base depth. |
| Aspen Snowmass | SNOTEL 542:CO Independence Pass | 32.6 km. Currently used as base depth. |
| Sun Valley | SNOTEL 537:ID Hyndman | 20.3 km. Currently used as base depth; replaced below. |
| Schladming | GeoSphere 11268 Dachstein glacier | 2,520 m, 505 m above the 2,015 m summit. Replaced below. |
| Lenzerheide | SLF PMA2 Colms da Parsonz | 16.4 km. Currently used as base depth; replaced below. |
| Big Sky, Jackson Hole, Snoqualmie, Whitefish, Mammoth, Mt Bachelor, Stevens Pass (2nd), Palisades (3rd), Vail (3rd) | various SNOTEL | 17 to 55 km. Most are display-only; the resorts keep OnTheSnow or a closer station. |

`scripts/validate_forecasts_v2.py` line 147 checks `distance_km > 60`; change it to 15.

### 0.2 Depth priority follows config order, not distance

`assemble_resort` sorts station rows by distance (line ~2017) before `choose_depth`, so today the nearest station with a value always wins. Several mappings below deliberately prefer a slightly farther station (a resort-village station over a ridge-top one). Keep a private `_rank` (the index in the resort's station config) on each row, have `from_station` pick the lowest `_rank` with a fresh value, and strip `_rank` before writing JSON. The output stays sorted by distance, so the validator rule on ordering still holds.

### 0.3 Depth chain per country

`choose_depth` today has no station step for Japan, France or Spain. New chains:

| Country | Chain |
|---|---|
| USA | OnTheSnow (≤3 days old) → SNOTEL (unchanged) |
| Austria | GeoSphere / Tirol station in config order |
| Switzerland | SLF station in config order |
| Japan | OnTheSnow → tenki.jp → JMA station |
| France | OnTheSnow → Météo-France station |
| Spain | OnTheSnow → Meteocat / Météo-France station |

Resort reports stay first where they exist because they describe the slopes. A station value is shown with its own elevation and distance in `source_label`, so a 1 cm valley reading is never passed off as a slope depth.

### 0.4 Staleness

A station reading older than its limit is kept in the station row (with its `observed_utc`) but is not used for `depth` or `recent.measured`; add a note like "Les Arcs reading is 7 days old, not used". The limits follow each source's update cycle with one missed update allowed:

| Source | Updates | Max age for depth |
|---|---|---|
| JMA AMeDAS | hourly snapshot | 6 h |
| Météo-France hourly file, automatic station | one file a day at ~06:50 UTC, hourly rows to ~04 UTC | 36 h |
| Météo-France hourly file, manual ski-patrol station (IDs ending 4xx) | once a day while the resort runs | 72 h |
| Meteocat XEMA | every 30 min, ~1 h lag | 12 h (Ulldeter only reports about 7 times a day) |
| Land Tirol OGD | file rebuilt hourly | 6 h |
| GeoSphere TAWES | 10 min | 6 h |
| SLF IMIS | 30 min | 6 h |
| SLF study plot | once a day at ~06 UTC | 36 h |
| SNOTEL | daily | 2 days (existing behaviour, date only) |

Put these in one dict, `STATION_MAX_AGE_H = {...}`, keyed by provider plus a station-type suffix where needed.

---

## 1. Sources and resort mappings by country

### 1.1 Japan: JMA AMeDAS (new provider `JMA`)

**Source.** Japan Meteorological Agency automated stations. Keyless. Licence: Public Data Terms of Use v1.0 (公共データ利用規約 第1.0版), reuse allowed with credit. Credit line for the sources block: `出典：気象庁ホームページ (https://www.jma.go.jp/)`, plus "processed by Snowforecast" because we pick stations and sum values. robots.txt is 404.

**Calls.**

1. `GET https://www.jma.go.jp/bosai/amedas/data/latest_time.txt` → e.g. `2026-11-27T20:30:00+09:00`.
2. Round down to the hour and fetch `https://www.jma.go.jp/bosai/amedas/data/map/YYYYMMDDHH0000.json`. Snow keys appear only in the on-the-hour files; the :10 to :50 files omit them. Checked today: `map/20261127200000.json` had `snow` on 336 stations, Kutchan `[6, 0]`, Hakuba `[1, 0]`.
3. Fetch the same hour 24 h and 48 h earlier (`...2026112620...`, `...2026112520...`) for new-snow totals. Checked: files 72 h back return 200.
4. Station metadata, once per run: `https://www.jma.go.jp/bosai/amedas/const/amedastable.json` (lat/lon as `[deg, minutes]`, `alt` in m, `enName`). Use `lat[0] + lat[1]/60`.

**Parsing.** Each value is `[value, quality_flag]`; use it only when the flag is `0`. `snow` is depth in cm. `snow24h` is new snow over the past 24 h in cm (JMA's own snowfall figure, built from depth increases). The timestamp in the filename is JST; `observed_utc` = filename time minus 9 h. If the on-the-hour file is missing, step back one hour, up to 3 times.

Fallback if the JSON layout changes: `https://www.data.jma.go.jp/stats/data/mdrr/snc_rct/alltable/snc00_rct.csv` (Shift-JIS, column 10 is current depth, quality code 8 = normal) and `snd24h00_rct.csv` for 24 h snowfall. Both verified today with the same values. Not needed for the first version.

**Mapping.** Distances checked against `forecasts_v2.json`.

| Resort | Station (id) | Distance | Station elev | Base / summit | vs base | 27 Nov reading |
|---|---|---|---|---|---|---|
| Niseko United | Kutchan (16217) | 6.3 km | 176 m | 260 / 1,200 | −84 m | 6 cm |
| Furano | Furano (12626) | 8.3 km | 174 m | 245 / 1,074 | −71 m | 0 cm |
| Rusutsu | Kimobetsu (16286) | 7.4 km | 264 m | 400 / 994 | −136 m | 10 cm |
| Hakuba Valley | Hakuba (48141) | 2.8 km | 703 m | 760 / 1,831 | −57 m | 1 cm |
| Happo-One | Hakuba (48141) | 3.1 km | 703 m | 760 / 1,831 | −57 m | 1 cm |
| Nozawa Onsen | Nozawa Onsen (48031) | 1.2 km | 576 m | 565 / 1,650 | +11 m | 0 cm |
| Shiga Kogen | Kusatsu (42121) | 12.4 km | 1,223 m | 1,300 / 2,307 | −77 m | 0 cm |
| Myoko Kogen | Shinano-Machi (48061) | 10.5 km | 685 m | 731 / 1,855 | −46 m | 0 cm |

Notes: these are town stations at or just below the base. Furano, Kiroro and Hakuba Valley keep OnTheSnow first, so the JMA row is shown under stations but does not replace the resort report (Hakuba's report says 124 cm higher up; the valley floor says 1 cm, and both are true). Shiga Kogen's station is across the Shirane ridge on the Gunma side, and its stored coordinates are rounded (36.7, 138.5). Myoko's station is over the prefecture line in Nagano; Sekiyama (54816, 350 m) is closer to the resort side but 381 m below base, which fails the elevation rule.

**Unresolvable in Japan.**

- Kiroro: the only snow station within 15 km is Akaigawa (16126, 13.8 km, 148 m), 372 m below the 520 m base. It fails the elevation rule. Kiroro keeps its OnTheSnow depth.
- Zao Onsen: the only snow station within 15 km is Yamagata city (35426, 11.6 km, 153 m), 627 m below base in a basin city. No depth.
- Niseko's lift-status feed (Yukiyama, `api.yukiyama.biz`) has no public terms and was not used.

### 1.2 France: Météo-France hourly climatological files (new provider `Meteo-France`)

**Source.** Météo-France "Données climatologiques de base – horaires" on data.gouv.fr (dataset `6569b4473bedf2e7abad3b72`). Keyless. Licence Ouverte 2.0 (`lov2` in the dataset API), credit "Météo-France". One source covers both the automatic stations and the manual ski-patrol (nivo) stations, because the manual observations are mirrored into these files under 8-digit IDs ending in `4xx` (checked: Les Menuires 73257400, Courchevel 73227400, Ceillac 05026400 all present with the same values as the nivo archive). So the first version needs only the hourly files; the two exceptions are noted under the mapping table.

**Calls.**

1. Resolve URLs once per run: `GET https://www.data.gouv.fr/api/1/datasets/6569b4473bedf2e7abad3b72/`, then pick the resource whose `url` matches `/BASE/HOR/H_{DEP}_latest-` for each department. Today that gives `https://meteofrance.s3.sbg.io.cloud.ovh.net/data/synchro_ftp/BASE/HOR/H_{DEP}_latest-2025-2026.csv.gz`. The season in the name rolls over (next is `latest-2026-2027`), so do not hard-code it. If the API call fails, fall back to the hard-coded 2025-2026 pattern.
2. Download departments `73`, `74`, `38`, `05`, `06`, plus `66` for La Molina in Spain. Each is 14 to 21 MB gzipped.

**Parsing.** Semicolon CSV, header row. Columns used: `NUM_POSTE`, `NOM_USUEL`, `LAT`, `LON`, `ALTI`, `AAAAMMJJHH` (UTC hour), `NEIGETOT` (total depth, cm), `HNEIGEF` (fresh snow, cm, manual stations only). Stream-decompress with `gzip.open(..., "rt")` and `csv.reader`, and keep only rows whose `NUM_POSTE` is configured and whose `AAAAMMJJHH` is within the last 80 hours; this keeps memory small. Per station, take the latest row with a non-empty `NEIGETOT` (not just the latest row: Flaine's last non-null value is 26 Nov 07 UTC with blanks after). `observed_utc` = that hour. A value of 0 is a real zero (Megève reports 0 cm), not missing. The file is republished once a day around 06:50 UTC, so the 00, 06, 12 and 18 UTC runs will mostly reuse the same data; cache by the resource's `last_modified` if download time matters.

**Mapping.** "Primary" is used first; "fallback" only when the primary is missing or stale.

| Resort | Role | Station (id) | Distance | Station elev | Base / summit | 27 Nov reading |
|---|---|---|---|---|---|---|
| Val Thorens | primary | VAL THORENS_SAPC (73257005) | 0.5 km | 2,277 m | 2,300 / 3,230 | 78 cm, 27 Nov 04Z |
| Les Arcs | primary | Les Arcs, manual (73054401) | 6.3 km | 2,040 m | 1,200 / 3,226 | 60 cm on 20 Nov (stale) |
| Les Arcs | fallback | LA PLAGNE_SAPC (73150004) | 8.0 km | 2,120 m | | 99 cm, 27 Nov 04Z |
| Val d'Isère | primary | VAL D'ISERE JOS (73304005) | 0.4 km | 1,850 m | 1,850 / 3,197 | 57 cm, 27 Nov 04Z |
| Les Menuires | primary | Les Menuires, manual (73257400) | 1.0 km | 1,800 m | 1,850 / 2,850 | 38 cm, 23 Nov 06Z |
| Les Menuires | fallback | VAL THORENS_SAPC (73257005) | 4.9 km | 2,277 m | | 78 cm |
| Avoriaz | primary | Avoriaz, manual (74191406) | 0.7 km | 1,768 m | 1,800 / 2,466 | none since 17 Apr |
| La Plagne | primary | LA PLAGNE_SAPC (73150004) | 0.7 km | 2,120 m | 1,250 / 3,080 | 99 cm, 27 Nov 04Z |
| La Plagne | second row | La Plagne, manual (73150400) | 0.1 km | 1,980 m | | 81 cm, 26 Nov 06Z, 6 cm fresh |
| Courchevel | primary | Courchevel, manual (73227400) | 0.3 km | 1,780 m | 1,300 / 2,738 | 70 cm, 23 Nov 06Z |
| Courchevel | fallback | Meribel Mottaret, manual (73015400) | 6.3 km | 1,700 m | | 35 cm, 26 Nov 06Z |
| Chamonix | primary | CHAMONIX (74056001) | 0.9 km | 1,042 m | 1,035 / 2,765 | 1 cm, 27 Nov 00Z |
| Morzine | primary | Les Gets, manual (74134400) | 4.1 km | 1,530 m | 1,000 / 2,000 | none since 29 Mar |
| Morzine | fallback | Avoriaz, manual (74191406) | 5.5 km | 1,768 m | | none since 17 Apr |
| Tignes | primary | TIGNES_SAPC (73296005) | 0.2 km | 2,084 m | 1,550 / 3,456 | 95 cm, 27 Nov 04Z |
| Méribel | primary | Meribel Mottaret, manual (73015400) | 3.5 km | 1,700 m | 1,100 / 2,952 | 35 cm, 26 Nov 06Z |
| Megève | primary | MEGEVE_NEIGE (74173402) | 0.8 km | 1,080 m | 1,113 / 2,353 | 0 cm, 26 Nov 06Z |
| La Clusaz | primary | LE GRAND-BORNAND (74136005) | 8.5 km | 1,430 m | 1,050 / 2,485 | 45 cm, 27 Nov 04Z |
| Flaine | primary | FLAINE_SAPC (74014004) | 0.7 km | 1,611 m | 1,600 / 2,480 | 42 cm, 26 Nov 07Z |
| Les Gets | primary | Les Gets, manual (74134400) | 1.2 km | 1,530 m | 1,172 / 1,826 | none since 29 Mar |
| Alpe d'Huez | primary | ALPE-D'HUEZ (38191002) | 5.9 km | 1,860 m | 1,125 / 3,330 | 68 cm, 27 Nov 04Z |
| Les 2 Alpes | primary | Les 2 Alpes Toura NE, manual (38253400) | 4.2 km | 2,550 m | 1,280 / 3,600 | none since 3 May |
| Les 2 Alpes | fallback | ST CHRISTOPHE EN OISANS_SAPC (38375009) | 8.5 km | 1,564 m | | 35 cm, 27 Nov 04Z |
| Serre Chevalier | primary | SERRE CHE-CHANTEMERLE, manual (05133400) | 1.8 km | 1,905 m | 1,200 / 2,800 | none since 18 Apr |
| Serre Chevalier | fallback | VILLAR ST PANCRACE (05183001) | 8.8 km | 1,310 m | | 10 cm, 27 Nov 03Z |
| Vars | primary | CEILLAC_NIVO, manual (05026400) | 11.0 km | 1,665 m | 1,650 / 2,750 | 30 cm, 26 Nov 06Z |
| Isola 2000 | primary | ISOLA 2000 (06073005) | 0.8 km | 1,912 m | 1,800 / 2,610 | 26 cm, 27 Nov 03Z |

Two resort-owned manual stations are missing from the hourly files and exist only in the separate nivo archive (`https://meteofrance.s3.sbg.io.cloud.ovh.net/data/OBS/NIVOSE/nivo_{YYYY}.csv.gz`, depth `ht_neige` in metres): LA CLUSAZ (nivo 7887, 1,965 m, 4.8 km) and VARS_NIVO (nivo 7929, 2,222 m, 4.0 km). Both are silent until their resorts open, and both resorts are already covered by the fallbacks above, so the archive is left out of the first version. Adding it later is a small optional step (one 330 KB file, same licence).

Notes: Les Arcs' own manual station last reported 7 days ago, so today the fallback La Plagne station (99 cm, 920 m above the listed base) is what shows. The listed bases for Les Arcs (1,200), La Plagne (1,250), Méribel (1,100) and Courchevel (1,300) are the lowest lift points; the stations sit in the main villages 400 to 900 m higher, which the elevation rule allows because they are inside the ski area. Chamonix's station is on the valley floor; Lognan (74056416, 1,970 m, 118 cm) is a slope option but 7.5 km away on Argentière, so it is not added. Tignes and Val Thorens keep OnTheSnow first while its report is under 3 days old.

**Not reporting today (seasonal).** Avoriaz, Morzine and Les Gets have stations configured, but the manual posts stop in spring and restart when the resorts open (last winter: Avoriaz 14 Dec, Les Gets 21 Dec). No automatic station within 15 km reports depth (Le Pleney 74191003 has been silent since 9 Oct). These three show "not reported yet" until then. No code change is needed when they restart.

### 1.3 Spain: Meteocat XEMA (new provider `Meteocat`) and Météo-France department 66

**Source A: Meteocat XEMA** via the Catalan open-data portal (Socrata). Keyless. Generalitat de Catalunya open-data terms, attribution "Servei Meteorològic de Catalunya". robots.txt asks for a 1-second crawl delay and does not block `/resource/`.

Call, one request for all configured stations:

```
GET https://analisi.transparenciacatalunya.cat/resource/nzvn-apee.json
    ?codi_variable=38
    &$where=codi_estacio in('Z1','YT','Z2','ZC') AND data_lectura >= '{now_utc - 80h, ISO, no Z}'
    &$order=data_lectura DESC
    &$limit=5000
```

Variable 38 is snow depth on the ground in cm, every 30 minutes. Fields: `codi_estacio`, `data_lectura` (UTC, e.g. `2026-11-27T10:30:00.000`), `valor_lectura` (string, convert to float). Station coordinates and altitude come from `https://analisi.transparenciacatalunya.cat/resource/yqwd-vj5e.json?$limit=1000` (`latitud`, `longitud`, `altitud`). Sanity-check that the newest `data_lectura` is not in the future; if it is, the timestamps are local and need converting. Low or bare sensors read 1 to 2 cm of noise; show the value as reported, do not round it to zero.

**Source B: Météo-France department 66**, the same hourly file and code as France (§1.2). `PUIGMAL-NIVOSE` (66067402) had 4 cm at 27 Nov 04Z in the hourly file. This is fresher than the daily file the Spain researcher used (2 days behind), so use the hourly one.

| Resort | Role | Station (id) | Distance | Station elev | Base / summit | 27 Nov reading |
|---|---|---|---|---|---|---|
| Baqueira-Beret | primary | Meteocat Bonaigua (Z1) | 6.1 km | 2,262 m | 1,500 / 2,610 | 14 cm, 10:30 UTC |
| Baqueira-Beret | fallback | Meteocat Bonabé (YT) | 10.2 km | 1,691 m | | 14 cm, 08:30 UTC |
| Boí Taüll | primary | Meteocat Boí (Z2) | 5.2 km | 2,537 m | 2,020 / 2,751 | 8 cm, 10:30 UTC |
| Vallter 2000 | primary | Meteocat Ulldeter (ZC) | 1.7 km | 2,413 m | 1,959 / 2,535 | 0 cm, 09:00 UTC |
| La Molina | primary | Météo-France PUIGMAL-NIVOSE (66067402) | 12.1 km | 2,461 m | 1,700 / 2,537 | 4 cm, 04:00 UTC |

Puigmal is on the French side of the massif. Bonaigua sits at the pass on the east edge of the Baqueira ski area.

**Unresolvable in Spain.**

- Formigal, Panticosa, Candanchú, Astún, Cerler (Aragon): no keyless station within 15 km. The nearest are French stations 19.5 to 24 km away. AEMET OpenData needs a key and returned nothing to check; SAIH Ebro needs an approved account. Neither is adopted (see §3).
- Sierra Nevada: no keyless live depth source found. AEMET only, unverified.
- Masella: Puigmal-Nivose and Meteocat Malniu are both 16.3 km away, just over the limit; Meteocat Das (1,096 m) is 504 m below base.
- Port Ainé: Meteocat Salòria is 16.7 km away; Sort (677 m) is about 1 km below base.
- Espot Esquí: Meteocat Espot (Z7, 4.6 km) is 2,519 m, about 400 m above the listed 2,110 m summit, so it fails the elevation rule. Bonaigua (12.8 km) passes the numbers but belongs to Baqueira across the valley, and both researchers called it a reference value at best, so it is not used. If the summit is confirmed near 2,500 m (one source says so), Z7 becomes valid with no other change.
- OnTheSnow currently gives no depth for any Spanish resort. That looks seasonal (resorts open in early December), not a broken source.

### 1.4 Austria: GeoSphere config swaps plus Land Tirol (new provider `Tirol`)

**Root cause of today's gaps.** The five GeoSphere stations configured for St. Anton, Lech (both 11110 Galzig), Sölden (11318 Brunnenkogel), Ischgl (11312 Galtür) and Zell am See (11340 Schmittenhöhe) have no snow-depth sensor; every `SCHNEE` value in the past 72 h is null. Obertauern's own station (11149) has none either, which is why the 49.7 km Flattnitz station was configured.

**Source A: GeoSphere TAWES**, already in the fetcher, unchanged call. Only the station IDs in `AUSTRIA_SKI_AREAS` change. Verified readings at 27 Nov 11:00 UTC: 11311 = 8 cm, 11308 = 18 cm, 11366 = 13 cm, 11351 = 13 cm.

**Source B: Land Tirol weather-station open data** (avalanche service LWD Tirol and hydrographic service HD Tirol). Keyless. CC BY 4.0, listed on data.gv.at as "Wetterstationsdaten Tirol"; credit "Land Tirol – data.tirol.gv.at".

```
GET https://wiski.tirol.gv.at/lawine/produkte/ogd.geojson
```

Fetch the `wiski.tirol.gv.at` address directly. The catalogue link on `lawine.tirol.gv.at` redirects there, and `lawine.tirol.gv.at/robots.txt` says `Disallow: /`, while `wiski.tirol.gv.at` has no robots.txt. Verified today: HTTP 200, 223 features.

Parsing: GeoJSON `features[]`; coordinates are `[lon, lat, elevation_m]`; properties `HS` (depth cm), `HSD24` / `HSD48` / `HSD72` (depth change cm), `date` (local time with offset, e.g. `2026-11-27T12:20:00+01:00`; convert to UTC), `name`, `operator`, `LWD-Nummer`. HD stations have no `LWD-Nummer`, so key on `LWD-Nummer` when present and on the feature `id` otherwise. Covers Tirol only, not Vorarlberg or Salzburg.

| Resort | Role | Station (id) | Dist. from stored coords | Dist. from real village | Station elev | Base / summit | 27 Nov reading |
|---|---|---|---|---|---|---|---|
| St. Anton | primary | GeoSphere ST.ANTON/ARLBERG (11311) | 2.8 km | 0.2 km | 1,304 m | 1,304 / 2,811 | 8 cm |
| St. Anton | extra row | Tirol Kapall (SKAP2) | 2.9 km | 2.1 km | 2,098 m | | 51.1 cm |
| Lech am Arlberg | primary | GeoSphere WARTH (11308) | 11.1 km | 6.3 km | 1,478 m | 1,444 / 2,811 | 18 cm |
| Lech am Arlberg | fallback | GeoSphere LANGEN/ARLBERG (11307) | 7.4 km | 8.6 km | 1,221 m | | 21 cm |
| Sölden | primary | Tirol HD "Sölden" (`80b5d623-5fb3-4abe-ed2e-ad5cfe789785`) | 12.7 km | 0.6 km | 1,382 m | 1,350 / 3,340 | 22.0 cm |
| Sölden | extra row | Tirol Giggijoch (SGIG2) | 10.4 km | 3.2 km | 2,532 m | | 46.1 cm |
| Ischgl | primary | Tirol Pischgraben (IPIS2) | 8.4 km | 2.1 km | 2,292 m | 1,377 / 2,872 | 46.3 cm |
| Ischgl | fallback | Tirol Nachtweide (INAC2) | 9.3 km | 3.4 km | 2,030 m | | 48.8 cm |
| Zell am See | primary | GeoSphere SAALBACH (11366) | 9.4 km | | 975 m | 757 / 2,000 | 13 cm |
| Schladming | primary | GeoSphere RAMSAU/DACHSTEIN (11351) | 4.8 km | | 1,207 m | 750 / 2,015 | 13 cm |

Notes: Warth is part of the linked Ski Arlberg area. Saalbach is the next valley over, not the Schmittenhöhe itself. The Kapall and Giggijoch rows are optional on-mountain context; they are not needed for coverage.

**Coordinate fixes (separate change, affects forecasts too).** Ischgl's stored point (46.968, 10.186) is Galtür; Ischgl village is about (47.012, 10.293). Sölden's stored point (46.913, 10.862) is the Brunnenkogel by the Pitztal glacier; the village is about (46.966, 11.008). Schladming's stored point sits on the Dachstein glacier, not the Planai. St. Anton's is about 3 km west of the village (47.130, 10.268) and Lech's about 6 km south of the village (47.208, 10.142). Every mapping above passes the 15 km rule with either the stored or the corrected coordinates, so the depth work does not wait on this. Fixing them will move the forecast grid points, so do it as its own change and re-run the forecast checks.

**Unresolvable in Austria.** Obertauern: GeoSphere 11149 Obertauern has no snow sensor and nothing else with depth is within 15 km. The Salzburg avalanche service has stations at the pass, but LAWIS only publishes chart images, not numbers. Obertauern goes empty when the 49.7 km Flattnitz station is removed.

### 1.5 Switzerland: SLF (existing provider), two fixes

- **Lenzerheide** loses PMA2 (16.4 km) under the 15 km rule. Replace it with **ROT3 Plang Bi** (3.9 km, 2,495 m, base 1,229 / summit 2,865), which has HS in the current bulk feed (42.8 cm). It is the same station Arosa uses; the two areas are linked.
- **Mürren**: configured SCH2 Türliboden is silent (no rows in the bulk feed; per-station endpoint returns 404 `NO_DATA`). Configure, in order:
  1. SCH2 (keep, in case it comes back),
  2. **SLF study plot 1MR0 Mürren** (4.2 km, 1,650 m, at base; 35 cm and 11 cm new snow at 27 Nov 06 UTC), from `https://measurement-api.slf.ch/public/api/study-plot/measurements` (fields `station_code`, `measure_date`, `HS`, `HN_1D`) with metadata from `/public/api/study-plot/stations` (`code`, `label`, `lat`, `lon`, `elevation`),
  3. **IMIS LHO2 Russisprung** (8.8 km, 2,150 m, 31.97 cm), already in the bulk feed the fetcher downloads.

  The study plot is a second, small endpoint on the same API and licence, so the provider stays `SLF`. Store a station type (`imis` or `study_plot`) on the config tuple to pick the endpoint and the staleness limit. If you want the smallest possible first step, ship LHO2 alone first (config only) and add the study plot later.

### 1.6 USA: SNOTEL (existing provider), config changes only

Verified against the AWDB station list (919 active SNOTEL sites with depth) and today's data call.

| Resort | Change | Station | Distance | Station elev | Base / summit | 26 Nov depth |
|---|---|---|---|---|---|---|
| Telluride | add (none today) | 1344:CO:SNTL Alta Lakes | 6.1 km | 3,441 m | 2,659 / 3,831 | 68.6 cm |
| Telluride | add as second | 538:CO:SNTL Idarado | 11.9 km | 2,981 m | | 22.9 cm |
| Big Sky | replace 754:MT (38 km) | 590:MT:SNTL Lone Mountain | 2.4 km | 2,688 m | 2,073 / 3,403 | 22.9 cm |
| Jackson Hole | replace 497:WY (41.8 km) | 689:WY:SNTL Phillips Bench | 10.4 km | 2,490 m | 1,924 / 3,185 | 25.4 cm |
| Sun Valley | replace 537:ID (20.3 km) | 895:ID:SNTL Chocolate Gulch | 9.8 km | 1,923 m | 1,753 / 2,789 | 15.2 cm |
| Park City | add before 628:UT | 814:UT:SNTL Thaynes Canyon | 3.8 km | 2,822 m | 2,103 / 3,056 | 48.3 cm |
| Kirkwood | add (none today) | 1067:CA:SNTL Carson Pass | 5.6 km | 2,548 m | 2,377 / 2,987 | 66.0 cm |
| Snoqualmie Pass | replace 788:WA (17.1 km) | 672:WA:SNTL Olallie Meadows | 5.6 km | 1,222 m | 945 / 1,652 | 2.5 cm |
| Aspen Snowmass | remove 542:CO (32.6 km) | none within 15 km (nearest 22.2 km) | | | | |

Telluride: Alta Lakes is first because it is on the ski-area side; Idarado is over the ridge in the Red Mountain drainage. SNOTEL triplets are checked against the AWDB name by `norm_name`, so the label in config must match the AWDB name exactly ("Alta Lakes", "Lone Mountain", "Phillips Bench", "Chocolate Gulch", "Thaynes Canyon", "Carson Pass", "Olallie Meadows", "Idarado").

**Unresolvable in the USA for a station:** Aspen Snowmass, Mammoth Mountain, Mt. Bachelor and Whitefish have no SNOTEL site within 15 km. Mammoth, Bachelor and Whitefish keep their OnTheSnow depth. Aspen's OnTheSnow report currently carries new snow but no depth, so Aspen's depth goes empty. California's CDEC network was rejected (robots.txt `Disallow: /`, and Mammoth Pass returned an impossible −187 in).

---

## 2. Measured new snow over 24, 48 and 72 hours

Today `recent.measured` is either the OnTheSnow report or a depth change at a station, and 58 resorts have no 48 h or 72 h figure. A depth change is not new snow: Davos shows −4.8 cm over 24 h while SLF's own measured new snow at the same station is 12.3 cm. So the plan adds a third kind and keeps depth change clearly labelled.

### 2.1 New kind

```json
"measured": {
  "kind": "measured_new_snow",
  "provider": "JMA",
  "label": "New snow measured at Kutchan station",
  "h24_cm": 0, "h48_cm": 0, "h72_cm": 4
}
```

Order in `recent_measured()`: OnTheSnow report (if fresh) → `measured_new_snow` from the highest-priority fresh station → `depth_change` from that station → `null`. Each total must come from consecutive, complete periods; if any day in the window is missing, that total is `null` (never assume a missing day was zero).

Add three keys to each station row so the station table can show them next to the depth change: `new_24h_cm`, `new_48h_cm`, `new_72h_cm` (null where the source has no new-snow figure). Add them to `STATION_KEYS` in the validator and a column to `render.py` screen 6.

### 2.2 Per source

| Source | 24 / 48 / 72 h figure | How | Kind |
|---|---|---|---|
| JMA | `snow24h` from the map files at T, T−24 h, T−48 h | h24 = snow24h(T); h48 = h24 + snow24h(T−24); h72 = h48 + snow24h(T−48). All three flags must be 0. | measured_new_snow |
| SLF IMIS | `GET /public/api/imis/daily-snow?period_in_days=7` (rows `station_code`, `measure_date`, `HS`, `HN_1D`; not sorted) | Sort by date; h24 = latest day's HN_1D; h48 and h72 sum the latest 2 and 3 consecutive days. Verified: WFJ2 HN_1D 12.3 cm today. | measured_new_snow |
| SLF study plot | `HN_1D` in study-plot measurements | h24 only from the current-day row; h48/h72 only if the endpoint returns earlier days (today it returns the latest day only, so leave them null). | measured_new_snow |
| Météo-France manual (4xx) | `HNEIGEF` on the daily 06 UTC row | Sum consecutive daily rows. | measured_new_snow |
| Météo-France automatic | `NEIGETOT` hourly history in the same file | Depth at latest hour minus depth 24/48/72 h earlier (nearest reading within 1 h). | depth_change |
| Meteocat | var 38 history from the same 80 h query | Same as above, nearest reading within 1 h. Daily new snow (var 1602, dataset `7bvh-jvq2`) exists but runs 2 days behind (latest is 25 Nov on 27 Nov), so it is not used for 24 h. | depth_change |
| Land Tirol | `HSD24`, `HSD48`, `HSD72` | Read directly. | depth_change |
| GeoSphere | `SCHNEE` history already fetched (72 h window) | Code only computes 24 h today; add 48 h and 72 h with the same `_closest_reading`. Widen the window to 74 h so T−72 h is inside it. | depth_change |
| SNOTEL | unchanged (15-day series already fetched) | Already computes 24/48/72 h depth change. | depth_change |

Result: 72 resorts with some measured recent figure (75 after the French season starts), and nearly all of them with 48 h and 72 h where the station reported through the whole window. Measured new snow (not depth change) becomes available from stations in Japan (8 resorts with a JMA station) and Switzerland (all 11), and at the French manual stations.

---

## 3. API keys

None of the adopted sources needs a key, so there is nothing to add to GitHub Actions now.

AEMET OpenData is the only lead for the six Aragon and Andalusia resorts, and it is not adopted because no value could be checked. If someone registers a free key and confirms that stations near Formigal, Candanchú/Astún, Panticosa, Cerler or Sierra Nevada actually report the `nieve` field, the pattern is:

- Repo secret `AEMET_API_KEY`, passed in `.github/workflows/refresh_v2.yml` as `env: AEMET_API_KEY: ${{ secrets.AEMET_API_KEY }}`.
- In the fetcher: `key = os.environ.get("AEMET_API_KEY")`; if empty, set `provider_status["AEMET"] = "skipped (no key)"`, add no warning, and leave the resorts empty with their existing note. Never log the key or the request URL that contains it.

---

## 4. Provider labels, `observed_utc` and the sources block

### 4.1 Provider strings

`sources_block` adds `s["provider"].lower()` to the used set and matches it against catalogue `id`s, so each provider string must lower-case to its catalogue id.

| Provider string | Catalogue id | Catalogue name | url | used_for |
|---|---|---|---|---|
| `JMA` | `jma` | Japan Meteorological Agency, AMeDAS | https://www.jma.go.jp/bosai/amedas/ | measured snow depth and new snow at Japanese stations (出典：気象庁ホームページ; selected and summed by Snowforecast) |
| `Meteo-France` | `meteo-france` | Météo-France, hourly climatological data (Licence Ouverte 2.0) | https://www.data.gouv.fr/datasets/6569b4473bedf2e7abad3b72/ | measured snow depth at French and Pyrenean stations |
| `Meteocat` | `meteocat` | Servei Meteorològic de Catalunya, XEMA | https://analisi.transparenciacatalunya.cat | measured snow depth at Catalan stations |
| `Tirol` | `tirol` | Land Tirol weather stations, LWD and HD Tirol (CC BY 4.0) | https://www.data.gv.at (Wetterstationsdaten Tirol) | measured snow depth at Tyrolean stations |

Also add the new names to `names` in `sources_block`, to `by_provider` in the run summary, and to `PROVIDERS`, the station-provider check (line 149) and `health()` (line 209) in `scripts/validate_forecasts_v2.py`.

### 4.2 `depth` and station rows

`from_station` already builds `source_label` as `"{provider} {name} station, {elev} m, {dist} km away"` and copies `observed_utc` from the row. Keep that. Rules for the new providers:

- `kind` = `"measured"` for every station value.
- `observed_utc` = the reading's time in UTC, ISO with `Z`, from: JMA filename time − 9 h; Météo-France `AAAAMMJJHH`; Meteocat `data_lectura`; Tirol `date` converted from its offset; SLF study plot `measure_date`.
- `observed_date` = the UTC date of the same reading.
- SNOTEL stays as it is: `observed_utc` null, `observed_date` set, because AWDB gives a date only.
- `series`: fill the 15-day series only where the data is already in hand (SNOTEL). Leave `[]` for the others rather than making extra calls.
- Station `name` comes from the provider's metadata, not from config, as today.

### 4.3 Staleness

See §0.4. When a value is too old, keep the row, leave `depth` to the next station or to null, and add a resort note naming the station and its age. The dashboard already prints the reading date from `observed_utc`.

---

## 5. Rollout order

Ranked by resorts gained per unit of work. Each step is independent and can ship alone.

| Step | Work | Resorts gained | Size |
|---|---|---|---|
| 1 | Acceptance rules (§0.1), config-order priority (§0.2), and config swaps in existing providers: GeoSphere 11311 / 11308 / 11366 / 11351, SLF ROT3 for Lenzerheide and LHO2 for Mürren, SNOTEL changes in §1.6, validator distance limit | +St. Anton, Lech, Zell am See, Mürren, Telluride (+5); −Aspen, −Obertauern (honest removals). Fixes four wrong-distance depths. | Small, mostly config |
| 2 | Météo-France hourly files (§1.2), depts 73, 74, 38, 05, 06, 66; France and Spain depth chains | +15 France now (+3 more in December), +1 La Molina | Medium: one parser, large downloads |
| 3 | JMA AMeDAS (§1.1), Japan chain | +6 Japan (Niseko, Rusutsu, Happo-One, Nozawa, Shiga, Myoko) | Small: four small JSON calls |
| 4 | Meteocat XEMA (§1.3) | +3 Spain (Baqueira, Boí Taüll, Vallter) | Small: one query |
| 5 | Land Tirol OGD (§1.4) | +2 Austria (Sölden, Ischgl) | Small: one GeoJSON |
| 6 | New snow (§2): SLF daily-snow, JMA snow24h sums, GeoSphere 48/72 h, Météo-France HNEIGEF and history, Meteocat and Tirol changes, new station keys, validator and render updates | 48/72 h figures for about 44 more resorts; measured new snow instead of depth change in Switzerland | Medium |
| 7 | SLF study plot for Mürren (§1.5) | Mürren moves from an 8.8 km ridge station to a station at its base | Small |
| 8 | Coordinate fixes for Ischgl, Sölden, Schladming, St. Anton, Lech (§1.4) | No coverage change; correct distances and forecast points | Small code, needs forecast re-check |

Steps 2 and 3 are close: step 2 gains more resorts, step 3 is less work. Either order is fine.

### Expected depth coverage after each step (today's data)

| After step | USA | AT | CH | JP | FR | ES | Total |
|---|---|---|---|---|---|---|---|
| Now | 21 | 5 | 10 | 3 | 2 | 0 | 41 |
| 1 | 21 | 7 | 11 | 3 | 2 | 0 | 44 |
| 2 | 21 | 7 | 11 | 3 | 17 | 1 | 60 |
| 3 | 21 | 7 | 11 | 9 | 17 | 1 | 66 |
| 4 | 21 | 7 | 11 | 9 | 17 | 4 | 69 |
| 5 | 21 | 9 | 11 | 9 | 17 | 4 | 71 |
| December, French resorts open | 21 | 9 | 11 | 9 | 20 | 4 | 74 |

### Still empty after all steps (12 today, 15 counting the seasonal French ones)

| Resort | Reason |
|---|---|
| Aspen Snowmass | No SNOTEL site within 15 km; OnTheSnow gives no depth |
| Obertauern | No station with a snow sensor within 15 km; Salzburg avalanche data is images only |
| Zao Onsen | Only snow station within 15 km is 627 m below base in Yamagata city |
| Formigal, Panticosa, Candanchú, Astún, Cerler | No keyless station within 15 km; AEMET needs a key and is unverified |
| Sierra Nevada | AEMET only, unverified |
| Masella | Nearest suitable stations are 16.3 km away |
| Port Ainé | Nearest suitable station is 16.7 km away |
| Espot Esquí | Nearby station is 400 m above the listed summit |
| Avoriaz, Morzine, Les Gets (until mid December) | Resort stations not reporting until opening day |

Kiroro is not in this list because it keeps its OnTheSnow depth; its only JMA station is too low to use.

---

## 6. Checks to run after each step

- `python3 scripts/fetch_data_v2.py` then `python3 scripts/validate_forecasts_v2.py` and `--health`.
- For each new mapping, confirm the station row's `distance_km` matches the tables here within 0.2 km, and that `observed_utc` is within the staleness limit.
- Spot-check one value per provider against the source on the same run (for example Kutchan in the JMA map file, TIGNES_SAPC in `H_73`, Z1 in Socrata, IPIS2 in the Tirol file).
- Confirm no resort's depth `source_label` names a station more than 15 km away.

---

## 7. Queue: publish resort open/closed status (fetcher item)

Added 27 Nov after the user's concern that "best ski day" makes no sense where we have no snow depth (ux spec rule 5c). Today the dashboard can only soften its claim on depth ("Ski Mon" becomes "Most snow Mon: 26 cm." under a 20 cm base). Depth is a weak proxy for "can I ski there": a station reading sits off-piste at its own elevation, and Sölden's glacier is open on a 22 cm reading. Whether the resort is open is the real signal, and the fetcher already reads it but throws it away.

### Where it is parsed today (scripts/fetch_data_v2.py)

- `OTS_STATUS_RE` (line 1752) matches `"status":{"openFlag":N,"openingDate":"YYYY-MM-DD"}` in the OnTheSnow page. The comment above it records openFlag 1 = open, 2 = closed (seen 27 Nov 2026: Tignes 1, Val d'Isère and Méribel 2).
- `parse_onthesnow()` (line 1755) stores the match on the report dict as `report["_open_flag"]` and `report["_opening_date"]` (lines 1770-1771).
- `onthesnow_all()` (line 1797) reads them at lines 1825-1829 (`flag`, `closed`, `opening`, `open_words`) only to word the notes and to drop a closed resort's placeholder zeros. Nothing is copied into `rep` / `r["ots_report"]`, so `data/forecasts_v2.json` has no status field.
- Caution: the status must be captured right after `parse_onthesnow()` returns (line 1809), before the three early `continue`s for "no report date" (line 1818), "report is stale" (line 1824) and "closed and all zeros" (line 1839). Note that `parse_onthesnow()` only looks for the status after a snow object, so a page with no snow object (line 1813) yields no status; that case stays null. A closed resort is exactly the case that hits the last of these, so reading it later would lose the most useful value.

### What to publish

One object per resort, `resort.status`:

```json
"status": {"open": true | false | null, "reopens": "YYYY-MM-DD" | null, "source": "OnTheSnow" | null, "observed_utc": "..." | null}
```

- `open`: true for openFlag 1, false for 2, null when the page states no status, the page failed, or the resort has no OnTheSnow page. Any other openFlag value is null plus a warning naming the value; never guess.
- `reopens`: `openingDate` when `open` is false and the date parses and is not in the past, else null.
- `source`: "OnTheSnow" when `open` is not null, else null. `observed_utc`: the run time of the fetch (the status is current as of the page load, not the snow-report date).
- No status is inferred from depth, snowfall or the calendar. Missing stays null.
- Add the field to `scripts/validate_forecasts_v2.py`: types as above, `reopens` null unless `open` is false.

### Display behavior once the field arrives (dashboard_v2/, not done yet)

Only `open is false` changes anything. `open: null` behaves exactly as today (rule 5c depth wording), and `open: true` enables the imperative even under 20 cm depth, because a resort that says it is open has a skiable base by its own report.

1. Resort page verdict: "Closed until Fri 5 Dec." when `reopens` is set and inside the forecast window or later; "Closed for the season." is not used (we cannot tell season-end from not-yet-open); with no date, "Resort closed; most snow Mon: 26 cm." The verdict never says "Ski" for a closed resort, whatever the depth.
2. Best ski day cell (All resorts and Compare tables): "— closed" in the answer form of rule 5 ("none — wind"), in place of the day and amount, when `open` is false and the best day falls before `reopens` (or there is no `reopens`). If the best day is on or after `reopens`, the cell keeps its day and amount and the verdict reads "Ski Sat; opens Fri 5 Dec." when the base rule allows it.
3. Chronological best-day sort (rule 5a): a "— closed" cell sorts with the reason-word rows, after every dated row. This is the one place status re-ranks; depth still never does.
4. Powder view: a resort with `open` false is excluded from both the fallen-snow and forecast lists for any day before `reopens`. Its reason shows in the debug panel as "closed" like the other gate reasons.
5. Ground line: unchanged. Depth and fallen snow are facts and stay visible for a closed resort.
6. Footer sources: "Open/closed status: OnTheSnow" once, when at least one resort has a non-null `open`.
