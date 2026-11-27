# Resort expansion: Asia

Written 27 Nov 2026 and updated the same day with a verification pass (Japan and Korea in one pass, China, Central Asia, the Caucasus, Turkey, Iran and India in another). Nothing in `scripts/fetch_data_v2.py` or `scripts/validate_forecasts_v2.py` has been changed yet. The Alps and US plan is in `docs/resort_expansion_plan.md`; the numbers for both plans together are in section 4.

Inputs: the scout results for Japan, Korea, China, Central Asia and the Caucasus, Turkey, Iran, Lebanon, Israel and India, then the verification pass. The 10 Japan resorts already covered (Furano, Hakuba Valley, Happo-One, Kiroro, Myoko Kogen, Niseko United, Nozawa Onsen, Rusutsu, Shiga Kogen, Zao Onsen) are excluded.

## Headline numbers

- Ready to fetch: 50 Asia resorts (Japan 14, South Korea 6, China 8, Kazakhstan 2, Georgia 2, Kyrgyzstan 1, Uzbekistan 1, Armenia 1, Azerbaijan 1, Russia 1, Turkey 5, Iran 3, Lebanon 1, Israel 1, India 3).
- The size rule originally dropped 11 verified rows. On 27 Nov 2026 the user ruled that all areas with verified data are welcome, so the 8 rows with a checked coordinate, base and summit (Tomamu, Kamui Ski Links, Tazawako, Thaiwoo, Genting Secret Garden, Sarıkamış, Auli, Solang Valley) moved back into the ready roster. The other 3 stay out for data problems, not size: Tochal (conflicting base), Xiling (unreliable source point), Tetnuldi (no verified base point). Details in section 1c.
- Still unverifiable: no whole rows. A few single values are still unchecked (section 1d).
- Both plans together: 220 resorts (86 today, 84 from the Alps and US plan, 50 from this plan). About 70 Open-Meteo requests a run and 280 a day. Counted the way Open-Meteo bills, that is roughly 10,750 to 12,850 calls a day, which is over the free tier's 10,000 (section 4).

## 1. Roster

| Country group | Scout rows | After dedupe | Ready to fetch | Held (data problem) | Needs verification |
|---|---|---|---|---|---|
| Japan | 15 | 14 | 14 | 0 | 0 |
| South Korea | 6 | 6 | 6 | 0 | 0 |
| China | 9 | 9 | 8 | 1 | 0 |
| Kazakhstan, Kyrgyzstan, Uzbekistan, Georgia, Armenia, Azerbaijan, Russia | 10 | 10 | 9 | 1 | 0 |
| Turkey, Iran, Lebanon, Israel, India | 14 | 14 | 13 | 1 | 0 |
| **Total** | **54** | **53** | **50** | **3** | **0** |

Japan's "after dedupe" went from 13 to 14 because Lotte Arai is now its own resort. "Ready to fetch" means the row has a checked base coordinate, an integer base and a summit, and is high or medium confidence. The "held" column was "dropped (size rule)" until the user's 27 Nov 2026 ruling restored the 8 size-dropped rows with verified data; the 3 still held (Xiling, Tetnuldi, Tochal) are out for data problems.

### 1a. Decisions applied in the verification pass

- **Lotte Arai is its own resort.** It is on Mt Okenashi, not Mt Myoko, is owned by Lotte, and has its own gondola, lifts and ticket. Its point is about 13 km from the existing Myoko Kogen point (36.88, 138.12), not 6 km as first thought. It has a passing JMA station (Sekiyama).
- **Kagura stays merged into Naeba,** but the merged entry now uses Naeba's own lift top (1,789 m). The 1,845 m Kagura-side summit was not re-verified.
- **Appi Kogen's base is 620 m.** The official resort site gives base 620 m, top 1,328 m and 708 m of vertical, and skiresort.info and ja.wikipedia agree. The 500 m figure is the old Zeiler base, whose gondola has been suspended since the 2019-20 season. At 620 m the Iwate-Matsuo station fails the elevation rule, so Appi has no JMA station.
- **Sapporo Teine moves to its true base.** The old point (43.084, 141.205, from en.wikipedia) sits at about 565 m on the upper Highland side. The base is 43.1043, 141.2076 (340 m). At the true base, the Koganeyu station is 15.7 km away and fails the 15 km rule, so Teine has no JMA station. This is open decision 1: keeping the old point keeps Koganeyu at 13.5 km, but Koganeyu is in the Jozankei valley, not on Teine's slope.
- **Beidahu uses the gondola base** (43.4172, 126.6131), about 300 m south of the town point. Two sources agree.
- **Tochal's ski area starts at 3,550 m, not 1,900 m.** 1,900 m is the Velenjak gondola bottom in Tehran, where there is no skiing. Tochal stays out because its base is conflicting (sources describe the gondola bottom and the ski area start as the base), not because of size.
- **Xiling's summit conflict was a misreading.** The ski area runs from 2,161 to 2,415 m; 1,408 m is the access cable car. It stays out because the only source point is the access cable car, 4 km from the ski area, so the coordinate is unreliable.
- **One size rule for every country — superseded for data-verified rows.** The rule was: an area with 7 or more main lifts stays, and an area with 3 to 6 lifts stays only if it is the biggest resort in its own country (main lifts from skiresort.com, without magic carpets). On 27 Nov 2026 the user ruled that all areas with verified data are welcome, which restored the 8 size-dropped rows with a checked coordinate, base and summit. The rule's text is kept for the record; it still describes why Goderdzi, Hatsvali, Tufandag and Beldersay were cut at scouting, before any data was verified.
- **Tomamu, Kamui and Tazawako were dropped by that rule and are now restored.** Each has 6 lifts and none is Japan's biggest resort, but their data was fully checked, so the user's ruling puts them back in. Open decision 2 is settled.
- Gala Yuzawa, Ishiuchi Maruyama and Joetsu Kokusai stay separate. Gala and Ishiuchi are lift-linked but have separate operators and tickets.

### 1b. Values that changed from the scout or skiresort.info figures

- Tomamu base 540 m (lowest lift) instead of 586 m.
- Sahoro base 420 m instead of 410 m.
- Ryuoo base 850 m instead of 869 m.
- Joetsu Kokusai summit 1,017 m (Mt Atema lift top, published spec) instead of 931 m.
- Madarao Kogen must not use skiresort.info: its page is the combined Madarao/Tangram area (800-1,340 m).
- Gala Yuzawa and Ishiuchi Maruyama must not use skiresort.info's combined Yuzawa Kogen/Gala/Ishiuchi entry (256-1,181 m).
- Naeba moves to the skiresort.info point (36.7931, 138.784), a slightly better base fix than the en.wikipedia point.
- Appi Kogen moves to the skiresort.info point (40.0009, 140.9715), about 180 m from the old point.
- Karakol summit 3,040 m. skiresort.com's 3,450 m is not lift-served in other sources.
- Wanlong base 1,550 m (two sources), so it no longer has a null base.
- Uludağ base 1,767 m (snow-forecast; skiresort.com gives 1,781 m), with the point moved to the Kuşaklıkaya chair in the hotel zone.

Several Wikipedia points are mid-mountain and must not be used as base points: Gala, Ishiuchi, Joetsu, Tazawako, Madarao, Appi (ja.wikipedia) and Phoenix (en.wikipedia).

### 1c. The size-rule drops, and what remains held (3)

The size rule originally dropped 11 verified rows here. The user's 27 Nov 2026 ruling ("all areas where we have data are welcome") restored the 8 with a checked coordinate, base and summit: Tomamu, Kamui Ski Links, Tazawako, Thaiwoo, Genting Secret Garden, Sarıkamış, Auli and Solang Valley. Their entries are now in section 2 with the rest of the roster. Their size-rule context, kept for the record: Tomamu, Kamui and Tazawako each have 6 lifts; Thaiwoo has 4 main lifts and Genting Secret Garden 5 (both in the Chongli cluster with Wanlong and Fulong — Genting shares a mountain and lift pass with Wanlong, 5.2 km away, and Thaiwoo is 6.4 km from Genting); Sarıkamış has 5 chairlifts; Auli has 3 lifts and about 3 km of slopes (2,702 m is the lowest lift; 2,520 m is the lowest slope); Solang Valley has 2 lifts and about 2.1 km (travel-site figures of 2,560 / 3,200 m do not match the terrain at the ropeway, so the verified 2,475 / 2,873 m pair is used).

These 3 stay out of the fetcher lists for data problems, not size:

| Name | Country | Lat, lon | Base / summit (m) | Why held |
|---|---|---|---|---|
| Xiling Snow Mountain | China | 30.7033, 103.1899 | 2,161 / 2,415 | Unreliable source point: skiresort.com's coordinate is the access cable car, 4 km east and 750 m below the ski area. No verified base coordinate. |
| Tetnuldi | Georgia | 43.0279, 42.8691 | 2,265 / 3,160 | No verified base point: Wikipedia's coordinate is the peak, and no base coordinate was sourced. |
| Tochal | Iran | 35.887, 51.4016 | 3,550 / 3,850 | Conflicting base: sources mix the 1,900 m Velenjak gondola bottom (no skiing) with the 3,550 m ski area start. If resolved, use 3,550 m. |

### 1d. Still unverified

No row is held back for missing data. These single values are still unchecked:

- Naeba: the 1,845 m Kagura-side summit. The entry uses 1,789 m, Naeba's own lift top.
- Sapporo Teine: which point to use (open decision 1).
- OpenStreetMap lift checks failed (server errors) for Kusatsu Onsen, Joetsu Kokusai, Phoenix Pyeongchang, Vivaldi Park and Tazawako. Those rows rest on the source coordinate plus the terrain model.
- Joetsu Kokusai: only one source places the coordinate at the base. The terrain model backs it (217 m against a 200 m base).
- Wanlong, Karakol and Uludağ coordinates rest on OpenStreetMap lift stations checked against terrain height, so they are medium confidence.
- Carried over from the first pass, not re-checked: Rosa Khutor's summit (2,320 m used; skiresort.info gives 2,509 m), Bakuriani's base (one secondary guide), and Alpensia's summit (775 / 970 m against SnowAction's 700 / 892 m).

### 1e. How rows were checked

Each base point came from skiresort.com's "at the ski lift" coordinate, the ja.wikipedia or Wikidata point, or the bottom station of an OpenStreetMap lift. Each was checked against SRTM 30 m terrain height (through opentopodata), because Open-Meteo's elevation endpoint had hit its daily limit. Elevations were checked against official resort sites, ja.wikipedia and en.wikipedia infoboxes, skiresort.info, SURF&SNOW, snow-forecast.com and Powderhounds. Wikipedia had no infobox coordinates for any of the Chinese resorts.

### 1f. Sanity check (python3, all 50 ready rows plus both plans)

Run on 27 Nov 2026 against the code block in section 2, the 86 rows the fetcher builds today, and the 84 main-roster rows in `docs/resort_expansion_plan.md`. The 8 restored rows were re-checked the same day (in-country box and base below summit all pass; Chongli distances below):

- All 50 Asia coordinates fall inside a rough box for their country.
- Base is below summit on every row in all three lists (220 rows).
- The tuple elevation matches the `RESORT_ELEVATIONS` base on every Asia row, and every elevation entry has a tuple.
- No name appears twice across the three lists (today, Alps and US, Asia).
- Six pairs sit within 3 km of each other. All are separate resorts that are neighbours: Yongpyong and Alpensia (1.8 km) and Shemshak and Darbandsar (3.0 km) in Asia, Solitude and Brighton (1.8 km) in the Alps and US plan, and three already in the fetcher (Hakuba Valley and Happo-One 0.3 km, Snowbird and Alta 1.2 km, Sölden and Pitztal Glacier 2.1 km). The last one looks odd, since Sölden and the Pitztal glacier are in different valleys; it is worth a look in the existing list.
- The restored Chongli rows cluster but stay apart: Genting Secret Garden is 5.2 km from Wanlong (they share a mountain and lift pass but are separate resorts with separate bases) and 6.4 km from Thaiwoo; Thaiwoo is 10.4 km from Wanlong.
- No Asia row is within 10 km of a resort already in the fetcher.

The box test cannot catch a coordinate that is in the right country but at the wrong resort. The terrain checks in 1e cover that.

### 1g. Country labels with a dispute

Mount Hermon is in the Golan Heights (Israel operates it, Syria claims it). Gulmarg is in Kashmir. Rosa Khutor is in Russia's Caucasus, and Russia is not clearly Asia. These are labelling choices, not data errors.

## 2. Fetcher-ready entries

Ready now: 50 rows (the 8 restored under the user's ruling are marked). Each has a checked base coordinate, an integer base and a summit. Timezones below are the IANA zones expected for each resort. The fetcher does not store a timezone; Open-Meteo's `timezone=auto` sets it from the coordinates, so these are checks, not inputs.

Format follows the existing lists. Area tuples are `(name, lat, lon, region, elevation)`. The fifth field is not read for the base (the base comes from `RESORT_ELEVATIONS`), so it carries the base value, as the Japan list does.

```python
# ---- JAPAN_SKI_AREAS (append) ---- tz Asia/Tokyo
("Naeba", 36.7931, 138.7840, "Niigata", 900),              # merged: Naeba + Kagura/Mitsumata/Tashiro
("Appi Kogen", 40.0009, 140.9715, "Iwate", 620),
("Sapporo Teine", 43.1043, 141.2076, "Hokkaido", 340),     # true base; see open decision 1
("Madarao Kogen", 36.8514, 138.2849, "Nagano", 910),
("Sahoro Resort", 43.1670, 142.8116, "Hokkaido", 420),
("Gala Yuzawa", 36.9511, 138.7993, "Niigata", 358),
("Ryuoo Ski Park", 36.7889, 138.4504, "Nagano", 850),
("Kusatsu Onsen", 36.6292, 138.5858, "Gunma", 1245),
("Joetsu Kokusai", 37.0300, 138.8276, "Niigata", 200),
("Ishiuchi Maruyama", 36.9785, 138.8026, "Niigata", 256),
("Lotte Arai Resort", 36.9908, 138.1801, "Niigata", 329),
("Tomamu", 43.0650, 142.6233, "Hokkaido", 540),            # restored (size rule superseded)
("Kamui Ski Links", 43.7048, 142.1879, "Hokkaido", 150),   # restored (size rule superseded)
("Tazawako", 39.7615, 140.7597, "Akita", 578),             # restored (size rule superseded)

# ---- JMA_STATIONS (append) ---- all pass the rule: within 15 km, base-300 to summit+300
"Madarao Kogen": [("48061", "Shinano-Machi")],    # 9.2 km, 685 m (same station as Myoko Kogen)
"Sahoro Resort": [("20356", "Shintoku")],         # 10.3 km, 178 m
"Gala Yuzawa": [("54841", "Yuzawa")],             # 1.4 km, 340 m
"Ryuoo Ski Park": [("48031", "Nozawa Onsen")],    # 13.7 km, 576 m
"Kusatsu Onsen": [("42121", "Kusatsu")],          # 1.5 km, 1,223 m (same station as Shiga Kogen)
"Joetsu Kokusai": [("54841", "Yuzawa")],          # 9.9 km, 340 m
"Ishiuchi Maruyama": [("54841", "Yuzawa")],       # 4.1 km, 340 m
"Lotte Arai Resort": [("54816", "Sekiyama")],     # 7.4 km, 350 m
"Tomamu": [("12691", "Ikutora")],                 # passes the rule (checked in the held-row pass)
"Kamui Ski Links": [("15161", "Fukagawa")],       # passes the rule (checked in the held-row pass)
# No JMA entry: Naeba (nearest Minakami 16.3 km, Yuzawa 16.7 km), Appi Kogen (Iwate-Matsuo
# fails at a 620 m base), Sapporo Teine (Koganeyu 15.7 km from the true base), Tazawako
# (no station within 17 km).
# If open decision 1 keeps the old Teine point (43.084, 141.205), add instead:
# "Sapporo Teine": [("14191", "Koganeyu")],      # 13.5 km, 230 m

# ---- DEPTH_COVERAGE_NOTES (append) ----
"Naeba": "No JMA snow station within 15 km (Minakami 16.3 km, Yuzawa 16.7 km)",
"Appi Kogen": "Iwate-Matsuo (9.7 km, 275 m) is more than 300 m below the 620 m base, so it is not used",
"Sapporo Teine": "Koganeyu is 15.7 km from the base, past the 15 km limit, so it is not used",
"Tazawako": "No JMA snow station within 17 km",

# ---- RESORT_ELEVATIONS (append) ----
"Naeba": (900, 1789, "skiresort.info, en.wikipedia and ja.wikipedia (900-1,789 m). Kagura-side 1,845 m not verified, not used"),
"Appi Kogen": (620, 1328, "appi.co.jp official (base 620 m, Nishimori 1,328 m); skiresort.info and ja.wikipedia agree. 500 m was the old Zeiler base"),
"Sapporo Teine": (340, 1023, "skiresort.info, en.wikipedia and ja.wikipedia (340-1,023 m); coords OSM lowest Olympia chair, SRTM 341 m"),
"Madarao Kogen": (910, 1350, "ja.wikipedia, snow-forecast, Powderhounds, SURF&SNOW (910-1,350 m); coords OSM No.1 chair bottom, SRTM 912 m. Not skiresort.info (combined with Tangram)"),
"Sahoro Resort": (420, 1030, "ja.wikipedia, Rakuten Travel, SURF&SNOW (base 420 m); skiresort.info 410 m; summit 1,030 m all sources"),
"Gala Yuzawa": (358, 1181, "ja.wikipedia and Powderhounds (358-1,181 m); coords OSM Dirijans gondola bottom, SRTM 350 m"),
"Ryuoo Ski Park": (850, 1930, "ja.wikipedia and SURF&SNOW (base 850 m); skiresort.info 869 m; summit 1,930 m all three"),
"Kusatsu Onsen": (1245, 1600, "skiresort.info and ja.wikipedia (1,245-1,600 m); coords ja.wikipedia/Wikidata Tenguyama base, SRTM 1,258 m"),
"Joetsu Kokusai": (200, 1017, "ja.wikipedia, niigata-kankou.or.jp, SURF&SNOW, famiski (200-1,017 m); skiresort.info 215-931 m from runs"),
"Ishiuchi Maruyama": (256, 920, "ja.wikipedia, niigata-kankou.or.jp, Weathernews, SURF&SNOW (256-920 m); coords OSM Sunrise Express bottom, SRTM 262 m"),
"Lotte Arai Resort": (329, 1280, "skiresort.info and ja.wikipedia (329-1,280 m); standalone resort on Mt Okenashi"),
"Tomamu": (540, 1171, "verification pass (540-1,171 m); 540 m is the lowest lift, skiresort.info's 586 m not used"),
"Kamui Ski Links": (150, 751, "verification pass (150-751 m)"),
"Tazawako": (578, 1186, "verification pass (578-1,186 m); ja.wikipedia's point is mid-mountain, not used as the base"),

# ---- KOREA_SKI_AREAS (new list, country "South Korea") ---- tz Asia/Seoul
("Yongpyong Resort", 37.645, 128.681, "Gangwon-do", 745),
("High1 Resort", 37.208, 128.826, "Gangwon-do", 717),
("Alpensia Resort", 37.661, 128.680, "Gangwon-do", 775),
("Muju Deogyusan Resort", 35.891, 127.737, "Jeollabuk-do", 745),
("Phoenix Pyeongchang", 37.5820, 128.3256, "Gangwon-do", 680),
("Vivaldi Park Ski World", 37.6454, 127.6825, "Gangwon-do", 282),
# RESORT_ELEVATIONS
"Yongpyong Resort": (745, 1450, "skiresort.com (base 745 m, top 1,450 m); Wikipedia base 770, top 1,458"),
"High1 Resort": (717, 1376, "skiresort.com (base 717 m, top 1,376 m); Wikipedia base 733"),
"Alpensia Resort": (775, 970, "skiresort.com (base 775 m); SnowAction 700 m base, 892 m top, within 100 m"),
"Muju Deogyusan Resort": (745, 1520, "Wikipedia (base 745 m, top 1,520 m); skiresort.com base 724 m"),
"Phoenix Pyeongchang": (680, 1080, "skiresort.info and snow-forecast (680-1,080 m); SnowAction 660-1,050; coords skiresort.info, SRTM 665 m"),
"Vivaldi Park Ski World": (282, 576, "skiresort.info daemyung-vivaldi-park (282-576 m); snow-forecast 310-580; SRTM 291 m"),

# ---- CHINA_SKI_AREAS (new list) ---- tz Asia/Shanghai
("Yabuli Ski Resort", 44.781, 128.450, "Heilongjiang", 457),
("Beidahu", 43.4172, 126.6131, "Jilin", 530),
("Changbaishan Wanda", 42.1021, 127.5050, "Jilin", 835),
("Songhua Lake", 43.6828, 126.6197, "Jilin", 330),
("Wanlong", 40.9666, 115.3722, "Hebei", 1550),
("Fulong", 40.9763, 115.2999, "Hebei", 1302),
("Thaiwoo", 40.8903, 115.4439, "Hebei", 1650),             # restored (size rule superseded)
("Genting Secret Garden", 40.9469, 115.4279, "Hebei", 1702),  # restored (size rule superseded)
"Yabuli Ski Resort": (457, 1375, "skiresort.com (base 457 m, top 1,375 m); Wikipedia base 397, top 1,345"),
"Beidahu": (530, 1405, "skiresort.com (530-1,405 m); coords OSM G gondola bottom, SRTM 530 m"),
"Changbaishan Wanda": (835, 1205, "skiresort.com (835-1,205 m); coords OSM G gondola bottom, SRTM 867 m"),
"Songhua Lake": (330, 900, "skiresort.com (330-900 m); coords OSM A gondola bottom, SRTM 332 m"),
"Wanlong": (1550, 2110, "skiresort.com and Skipedia (base 1,550 m), SkiAsia 1,560 m; top 2,110 m all three; coords OSM lift 8 bottom"),
"Fulong": (1302, 1720, "skiresort.com (1,302-1,720 m); coords OSM L1 bottom, SRTM 1,303 m"),
"Thaiwoo": (1650, 2160, "verification pass (1,650-2,160 m); Chongli cluster, 6.4 km from Genting Secret Garden"),
"Genting Secret Garden": (1702, 2100, "verification pass (1,702-2,100 m); shares a mountain and lift pass with Wanlong, 5.2 km away"),

# ---- CENTRAL_ASIA_CAUCASUS_SKI_AREAS (new list) ----
("Shymbulak", 43.12806, 77.08083, "Almaty region", 2260),         # tz Asia/Almaty
("Oi-Qaragai", 43.2294, 77.1343, "Almaty region", 1526),          # tz Asia/Almaty
("Karakol Ski Base", 42.4111, 78.4620, "Issyk-Kul", 2300),        # tz Asia/Bishkek
("Amirsoy Mountain Resort", 41.486706, 69.942341, "Tashkent region", 1630),  # tz Asia/Tashkent
("Gudauri", 42.47583, 44.47861, "Mtskheta-Mtianeti", 1993),      # tz Asia/Tbilisi
("Bakuriani", 41.75, 43.533, "Samtskhe-Javakheti", 1641),        # tz Asia/Tbilisi
("Tsaghkadzor", 40.53444, 44.69167, "Kotayk", 1966),             # tz Asia/Yerevan
("Shahdag Mountain Resort", 41.32167, 48.14806, "Qusar", 1435),  # tz Asia/Baku
("Rosa Khutor", 43.646, 40.332, "Krasnaya Polyana", 940),        # tz Europe/Moscow
"Shymbulak": (2260, 3180, "skiresort.info (base 2,260 m, top 3,180 m); Wikipedia 2,200 and 3,200"),
"Oi-Qaragai": (1526, 2140, "skiresort.com and Powderhounds (1,526-2,140 m; Aqtas gondola 2025); coords Umai chair bottom, SRTM 1,554 m"),
"Karakol Ski Base": (2300, 3040, "skiresort.com and central-asia.guide (base 2,300 m); snow-forecast and central-asia.guide (top 3,040 m)"),
"Amirsoy Mountain Resort": (1630, 2290, "Wikipedia (slopes 1,630 m, top 2,290 m); skiresort.info 1,630 to 2,286 m"),
"Gudauri": (1993, 3276, "skiresort.info (base 1,993 m, top 3,276 m); Wikipedia lowest lift 1,990, top 3,279"),
"Bakuriani": (1641, 2702, "secondary guide (base 1,641 m); J2ski confirms top 2,702 m; town 1,700 m"),
"Tsaghkadzor": (1966, 2820, "Wikipedia (base 1,966 m, top 2,820 m); skiresort.com top 2,819 m"),
"Shahdag Mountain Resort": (1435, 2525, "Wikipedia (base 1,435 m, top 2,525 m); a second source gives top 2,550 m"),
"Rosa Khutor": (940, 2320, "Wikipedia (base 940 m, top 2,320 m); OnTheSnow top 2,300 m; skiresort.info top 2,509 m, unresolved"),

# ---- TURKEY_IRAN_LEVANT_INDIA_SKI_AREAS (new list) ----
("Erciyes", 38.5419, 35.525, "Kayseri", 2088),                   # tz Europe/Istanbul
("Palandöken Ski Center", 39.86972, 41.27333, "Erzurum", 2200),  # tz Europe/Istanbul
("Kartalkaya", 40.66944, 31.78611, "Bolu", 1850),                # tz Europe/Istanbul
("Uludağ", 40.1023, 29.1405, "Bursa", 1767),                     # tz Europe/Istanbul
("Sarıkamış", 40.3225, 42.6182, "Kars", 2130),                   # tz Europe/Istanbul; restored (size rule superseded)
("Dizin", 36.04917, 51.41722, "Alborz", 2650),                   # tz Asia/Tehran
("Shemshak", 36.00875, 51.49472, "Alborz", 2550),                # tz Asia/Tehran
("Darbandsar", 36.021333, 51.465611, "Alborz", 2600),            # tz Asia/Tehran
("Mzaar Kfardebian", 34.01108, 35.82483, "Mount Lebanon", 1850), # tz Asia/Beirut
("Mount Hermon", 33.30806, 35.7725, "Golan Heights", 1600),      # tz Asia/Jerusalem
("Gulmarg", 34.04528, 74.38444, "Jammu and Kashmir", 2650),      # tz Asia/Kolkata
("Auli", 30.5368, 79.5654, "Uttarakhand", 2702),                 # tz Asia/Kolkata; restored (size rule superseded)
("Solang Valley", 32.3141, 77.1550, "Himachal Pradesh", 2475),   # tz Asia/Kolkata; restored (size rule superseded)
"Erciyes": (2088, 3346, "skiresort.com (base 2,088 m, top 3,346 m); Wikipedia base about 2,100"),
"Palandöken Ski Center": (2200, 3176, "Wikipedia (base 2,200 m, top 3,176 m); skiresort.info 2,148 and 3,140 for Ejder 3200"),
"Kartalkaya": (1850, 2255, "Wikipedia (base 1,850 m, top 2,255 m); skiresort.com 1,898 and 2,195"),
"Uludağ": (1767, 2322, "snow-forecast (base 1,767 m), skiresort.com 1,781 m; top 2,322 m both; coords OSM Kuşaklıkaya chair bottom"),
"Dizin": (2650, 3600, "Wikipedia, snow-forecast and damawand.de agree; skiresort.info 2,530 and 3,410"),
"Shemshak": (2550, 3050, "Wikipedia (base 2,550 m, top 3,050 m); skiresort.info 2,539 and 2,985"),
"Darbandsar": (2600, 3500, "Wikipedia (base 2,600 m, top 3,500 m); SummitPost 2,650 and 3,570"),
"Mzaar Kfardebian": (1850, 2465, "Official site, skiresort.com and Snowsure agree (1,850 m base, 2,465 m top)"),
"Mount Hermon": (1600, 2040, "Official site and Wikipedia agree (1,600 m base, 2,040 m top)"),
"Gulmarg": (2650, 3980, "skiresort.com and Wikipedia agree (2,650 m base, 3,980 m top); coordinates are the gondola point"),
"Sarıkamış": (2130, 2632, "verification pass (2,130-2,632 m)"),
"Auli": (2702, 3049, "verification pass (2,702-3,049 m); 2,702 m is the lowest lift, 2,520 m the lowest slope"),
"Solang Valley": (2475, 2873, "verification pass at the ropeway (2,475-2,873 m); travel-site 2,560/3,200 m figures not used"),
```

### Rules that every new entry also needs

- A `DEPTH_COVERAGE_NOTES` line for each resort that has no station. Without it the empty depth is unexplained. Every Asia resort outside the ten Japan rows with a JMA station needs one (40 rows).
- Country names added to `COUNTRIES` in `scripts/validate_forecasts_v2.py`. The current set is Japan, France, Spain, USA, Austria and Switzerland, and the Alps plan adds Italy, Germany and Slovenia. Asia adds South Korea, China, Kazakhstan, Kyrgyzstan, Uzbekistan, Georgia, Armenia, Azerbaijan, Russia, Turkey, Iran, Lebanon, Israel and India. Add only the countries that have entries in a phase that is running.
- The `base_depth` summary in `main()` lists countries by name. New countries will not appear in it until the list is extended.
- The validator default `expected_count=86` needs updating after each phase.
- `build_resorts()` loops over country tuples. Each new area list needs its own entry in that loop. Kyrgyzstan is new in this pass (Karakol), so the Central Asia list needs a country per row, not one country per list.

## 3. Data wiring

Nothing in Asia is wired yet except Japan's JMA station path, which already exists. Every other source below is a candidate. None has been checked for access, format or coverage unless marked as checked.

### Japan

- **Measured depth (JMA AMeDAS).** Checked. The fetcher's rule is a distance of at most 15 km and a station elevation between 300 m below the base and 300 m above the summit.
  - Pass (10): Madarao (Shinano-Machi), Sahoro (Shintoku), Gala, Ishiuchi and Joetsu (all Yuzawa), Ryuoo (Nozawa Onsen), Kusatsu (Kusatsu), Lotte Arai (Sekiyama), Tomamu (Ikutora), Kamui (Fukagawa).
  - No station (4): Naeba (nearest Minakami 16.3 km, Yuzawa 16.7 km), Appi Kogen (Iwate-Matsuo fails at the 620 m base), Sapporo Teine (Koganeyu 15.7 km from the true base; see open decision 1), Tazawako (no station within 17 km).
  - Shared stations: Yuzawa serves three new resorts. Shinano-Machi is also Myoko Kogen's station, Nozawa Onsen is Nozawa's, and Kusatsu is Shiga Kogen's. A shared station gives the same depth to each resort, which is correct for the rule but worth knowing when two resorts show the same number.
- **Resort reports (OnTheSnow).** No slugs for new Japan resorts have been checked. Each needs the same one-time redirect check the existing list had.
- **Resort reports (tenki.jp).** No IDs for new resorts. Each needs a checked page with the snow depth pattern.
- Japan's depth chain already runs report, then tenki.jp, then station. New Japan resorts use the same chain once entries exist.

### South Korea

- **Measured depth (KMA).** Candidate: the Korea Meteorological Administration open API (apihub.kma.go.kr). Unverified. It needs an auth key, which would be a repo secret like `AEMET_API_KEY`. A candidate station near Pyeongchang is Daegwallyeong. Unverified.
- **Resort reports.** None found that are open.
- **Launch state.** Forecast only until a key exists and a station passes the rule.

### China

- **Measured depth.** No open source identified. CMA open data has not been checked for snow depth at resort sites.
- **Resort reports.** None identified.
- **Launch state.** Forecast only.

### Central Asia and the Caucasus

- **Kazakhstan, Kyrgyzstan, Uzbekistan.** No open depth source identified. Forecast only.
- **Georgia (NEA), Armenia (Hydromet), Azerbaijan.** Candidate agencies. Unverified. Forecast only until checked.
- **Russia (Krasnaya Polyana).** No open source identified. Forecast only.

### Turkey, Iran, Lebanon, Israel, India

- **Turkey (MGM).** Candidate agency. Unverified. Forecast only.
- **Iran (IRIMO).** Public web sources only, and sanctions may limit data providers. Forecast only.
- **Lebanon.** No source identified. Forecast only.
- **Israel (IMS).** Candidate open data through data.gov.il for the Hermon area. Unverified. Forecast only until checked.
- **India (IMD).** Candidate agency. Unverified. Forecast only.

### What launches empty

At launch, 10 of the 50 Asia resorts have a measured depth (the ten Japan resorts with a JMA station). The other 40 are forecast only from Open-Meteo and need an empty-depth note.

## 4. Scale math, both plans together

This section covers today's roster, the Alps and US plan (`docs/resort_expansion_plan.md`, 84 main-roster rows) and this plan (50 rows, after the 8 restorations). It replaces the Asia-only figures from the first draft, which used a different weighting and did not match the Alps plan's measured baseline.

### How the fetcher calls Open-Meteo

From `scripts/fetch_data_v2.py`: three kinds of request, each batched 10 resorts at a time (`OM_BATCH_RESORTS = 10`).

- Daily: 3 points per resort (base, mid, summit), 15 variables, 16 forecast days plus 3 past days.
- Hourly: 3 points per resort, 17 hourly variables plus 8 current ones, 7 forecast days plus 3 past days.
- Models: summit only, one variable, 7 days. Batched separately for the alpine group (Austria and Switzerland, 3 models) and everyone else (2 models).
- About 3 more requests a run for the model check and retries (the Alps plan's measured figure).

The fetcher runs four times a day. Every resort in all three lists has a summit, so every one sends 3 points.

### Three ways to count

Open-Meteo's free tier allows 10,000 calls a day. How a multi-point request counts matters a lot here, so the table gives all three readings.

1. **Requests.** One HTTP request counts once.
2. **Points.** Each location in a request counts once.
3. **Weighted points (how Open-Meteo bills).** Each location counts once, and a location counts as more than one call when the request has more than 10 variables (variables divided by 10) or more than 14 days (days divided by 14). That makes a daily point about 2.0 calls (1.5 for 15 variables times 1.36 for 19 days) and an hourly point 1.7 to 2.5 calls (17 or 25 variables, depending on whether the current variables count). A models point is 1 call. This is my reading of Open-Meteo's published terms; confirm it on their pricing page before relying on it.

| Roster | Resorts | Requests per run | Requests per day | Points per day | Weighted calls per day | Share of 10,000 (weighted) |
|---|---|---|---|---|---|---|
| Today | 86 | 31 | 124 | 2,408 | 4,200 to 5,000 | 42% to 50% |
| Today + Alps and US | 170 | 55 | 220 | 4,760 | 8,300 to 9,900 | 83% to 99% |
| Today + Asia | 136 | 46 | 184 | 3,808 | 6,650 to 7,950 | 66% to 79% |
| **Today + both plans** | **220** | **70** | **280** | **6,160** | **10,750 to 12,850** | **108% to 129%** |

Requests per run for 220 resorts: 22 daily, 22 hourly, 23 models (5 alpine for the 43 Austria and Switzerland rows, 18 for the other 177), plus about 3. Retries on a 429 or a failed batch add to all of these.

What this means:

- If Open-Meteo counts requests or plain points, both plans fit easily (280 requests or 6,160 points a day).
- If it counts the way its terms describe, both plans together go over the free tier, and the Alps and US plan alone is already at the edge. Any manual or test run from the same IP adds on top. On 27 Nov 2026 at about 13:50, Open-Meteo's elevation endpoint returned "Daily API request limit exceeded" from this machine during verification, which fits this reading. The scheduled fetcher runs on GitHub's machines, so it probably does not share that limit, but any local fetcher run from this machine today would. Check the next scheduled run's log for 429 responses.
- At the weighted rate, one resort costs about 49 to 58 calls a day. The free tier holds about 170 to 200 resorts at today's settings.

Ways to fit 220 resorts under 10,000, none of them decided here:

- Cut the daily call to 14 days with no past days. The day factor drops from 1.36 to 1, which saves about 1,800 calls a day at 220 resorts and brings the total to roughly 9,000 to 11,050.
- Drop the mid point from the daily and hourly calls. That saves about a third of the point volume (about 3,300 to 4,000 a day at 220).
- Trim the hourly call to 10 variables or fewer, including the current block.
- Run three times a day instead of four, which takes the total to about 8,100 to 9,650.
- Pay for an Open-Meteo API plan, which also removes the non-commercial restriction.

### Runtime

The last full run took 685 seconds for 86 resorts (about 8 seconds a resort). A straight-line estimate:

| Roster | Resorts | Estimated run time |
|---|---|---|
| Today | 86 | 685 s (11.4 minutes, measured) |
| Today + Alps and US | 170 | about 1,354 s (22.6 minutes) |
| Today + Asia | 136 | about 1,085 s (18 minutes) |
| **Today + both plans** | **220** | **about 1,750 s (29 minutes)** |

The workflow timeout is 25 minutes today. The Alps plan raises it to 35 minutes in its first phase. At 29 minutes for both plans, 35 leaves about 6 minutes of slack, which a couple of slow retries (each Open-Meteo backoff waits up to 60 seconds) can use up. Raise it to 45 minutes if both plans ship. The estimate is rough: Asia adds 10 JMA lookups but no other station calls, so the per-resort cost for Asia is probably below the 8 second average.

## 5. Rollout

Each phase is one fetcher run that a person can check. Run with `--only` and a fresh `--out` path in the scratchpad, then run the validator on that output. Do not overwrite `data/forecasts_v2.json` until the phase passes. Before any Asia phase, settle the Open-Meteo budget question in section 4, because the Asia phases come after the Alps and US phases have already used most of the free tier.

**Phase J (Japan, 14 resorts).** The JMA path already exists.
1. Settle open decision 1 (Teine point).
2. Add the 14 Japan entries, the 10 JMA mappings and the 4 depth notes.
3. Check OnTheSnow and tenki.jp slugs and IDs for the 14 resorts.
4. Run `python3 scripts/fetch_data_v2.py --only "Naeba,Appi Kogen,Sapporo Teine,Madarao Kogen,Sahoro Resort,Gala Yuzawa,Ryuoo Ski Park,Kusatsu Onsen,Joetsu Kokusai,Ishiuchi Maruyama,Lotte Arai Resort,Tomamu,Kamui Ski Links,Tazawako"` and validate. Two Open-Meteo batches per kind.

**Phase K (South Korea, 6 resorts).**
1. Decide whether to get a KMA key. Without one, the phase is forecast only.
2. Add `South Korea` to `COUNTRIES`, the Korea list and its entries.
3. Run `--only "Yongpyong Resort,High1 Resort,Alpensia Resort,Muju Deogyusan Resort,Phoenix Pyeongchang,Vivaldi Park Ski World"` and validate.

**Phase C (China, Central Asia and the Caucasus, 17 resorts).** Yabuli, Beidahu, Changbaishan Wanda, Songhua Lake, Wanlong, Fulong, Thaiwoo, Genting Secret Garden, Shymbulak, Oi-Qaragai, Karakol Ski Base, Amirsoy, Gudauri, Bakuriani, Tsaghkadzor, Shahdag and Rosa Khutor.
1. Add the countries to `COUNTRIES`.
2. Add each resort's `DEPTH_COVERAGE_NOTES` line.
3. Run `--only` with those names and validate. Two Open-Meteo batches per kind.

**Phase T (Turkey, Iran, Lebanon, Israel, India, 13 resorts).** Erciyes, Palandöken, Kartalkaya, Uludağ, Sarıkamış, Dizin, Shemshak, Darbandsar, Mzaar, Hermon, Gulmarg, Auli and Solang Valley.
1. Add the countries to `COUNTRIES` and the depth notes.
2. Run `--only` with those names and validate.

Check: 14 + 6 + 17 + 13 = 50.

**Later.** The 3 rows still held in 1c (Xiling, Tetnuldi, Tochal) go in only once their data problems are resolved (a verified base point for Xiling and Tetnuldi, a settled base for Tochal). The unchecked values in 1d (Naeba's Kagura summit, Rosa Khutor's summit, Bakuriani's base) can be fixed in place without holding a phase.

## Open decisions

1. Sapporo Teine: the true base point (current choice, no JMA station) or the old mid-mountain point (keeps Koganeyu, 13.5 km).
2. ~~Tomamu, Kamui and Tazawako: dropped under the even size rule, or argued back in.~~ Settled 27 Nov 2026: the user ruled that all areas with verified data are welcome, so all three are in the ready roster.
3. Naeba: one merged entry with Naeba's own 1,789 m top (current choice), or verify Kagura's 1,845 m and use it.
4. Open-Meteo budget: which of the options in section 4 to take before both plans ship.
5. Whether Rosa Khutor belongs under Asia, given Russia's position.
