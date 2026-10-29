# Switzerland (CH): speed rules for the comma

Checked: 2026-10-29. Legal basis: Verkehrsregelnverordnung (VRV, SR 741.11, version in force 1 Oct 2026), Signalisationsverordnung (SSV, SR 741.21, version in force 1 Oct 2026), Federal Supreme Court rulings, ASTRA directives 15016, 15019 and 15010.

The upload `Swiss_Dynamic_Speed_Limits_SpeedVision_Review.md` (Italian) was checked against these sources. Its legal claims are correct; its scenarios, observation classes, safety rules and test cases are folded in below (sections "Variable message signs" and "For the comma").

## General limits (passenger cars, no sign)

| Road kind | Limit | Conditions | Source |
|---|---|---|---|
| Built-up area ("innerorts", densely built part of a town) | 50 km/h | Starts at sign 2.30.1 "50 generell", ends at 2.53.1. Also applies without a sign to drivers entering a town from unimportant side roads, once dense building starts. Holds in the whole densely built area, through junctions. | VRV art. 4a(1)(a), 4a(2); SSV art. 16(2), 22(3) |
| Outside built-up areas | 80 km/h | Applies from 2.53.1 or 2.53, or from 4.04 / 4.02 when leaving an Autostrasse / Autobahn. | VRV art. 4a(1)(b), 4a(3) |
| Autostrasse (expressway) | 100 km/h | From 4.03 to 4.04. The 100 must be signed on Autostrassen (SSV art. 22(5)). | VRV art. 4a(1)(c), 4a(3bis) |
| Autobahn (motorway) | 120 km/h | From 4.01 to 4.02. | VRV art. 4a(1)(d), 4a(4) |
| Tempo-30 zone | 30 km/h | Zone sign 2.59.1 to 2.59.2; only inside towns. | SSV art. 2a, 22a |
| Begegnungszone (meeting zone) | 20 km/h | Sign 2.59.5 to 2.59.6. Pedestrians have priority. | SSV art. 22b |
| Fussgängerzone (pedestrian zone) | walking pace | Sign 2.59.3 to 2.59.4. | SSV art. 22c |

All general limits apply only "unter günstigen Strassen-, Verkehrs- und Sichtverhältnissen" (VRV art. 4a(1)); art. 4(1) VRV requires stopping within the visible distance.

Conditions:

- **Wet road, fog, night:** no fixed number in the VRV. Only the general duty to adapt speed (art. 4 VRV). Not found: any rain or fog limit. Certainty high (VRV art. 4, 4a, 5 read in full).
- **Novice drivers:** no separate limit found in the VRV. Certainty medium (VRV searched; the driving-licence ordinance was not searched).
- **Vehicle type (not cars):** 80 km/h for heavy vehicles, trailer trains and cars with studded tyres; 100 km/h on Autobahn and Autostrasse for cars with a trailer up to 3.5 t (VRV art. 5). A car without a trailer is not affected.
- **Signed limits win** over the general ones (VRV art. 4a(5)).
- **Lawful signed values:** SSV art. 108(5) allows, in short: Autobahn below 120 (down to 60, 10 km/h steps); Autostrasse below 100 (down to 60); outside towns only below 80; in towns 80/70/60 or below 50; zones 30 and 20. So a signed number above the road's general limit is not lawful outside towns (art. 108 amended 12 Aug 2026, in force 1 Oct 2026, AS 2026 438). Certainty high.

## Road kinds and their signs

| Kind | Begins | Ends | Look |
|---|---|---|---|
| Autobahn | 4.01 "Autobahn" | 4.02 "Ende der Autobahn" | Green rectangle, white motorway symbol (road with overbridge); the end sign is the same with a red diagonal bar. |
| Autostrasse | 4.03 "Autostrasse" | 4.04 "Ende der Autostrasse" | Green rectangle, white front view of a car; the end sign has a red diagonal bar. |
| Town limit 50 | 2.30.1 "Höchstgeschwindigkeit 50 generell" | 2.53.1 "Ende der Höchstgeschwindigkeit 50 generell" | Round red-ringed 50 with "GENERELL" in white letters in the red ring; the end sign is grey/black with diagonal stripes. |
| Town (place name only) | 4.27 (main road, white on blue) / 4.29 (side road, black on white) | 4.28 / 4.30 (back of the same plate) | Rectangular place-name plate. |

Facts from the law:

- Autobahn and Autostrasse are roads reserved for motor traffic and signed as such. An Autobahn has a separate carriageway for each direction and no level crossings (VRV art. 1(3)). An Autostrasse can have a single carriageway (the definition does not require separation).
- 4.01 and 4.03 "heben alle zuvor signalisierten Beschränkungen auf" (lift all previously signed restrictions). 4.02 and 4.04 mean that the general rules apply again (SSV art. 45(1)).
- **Town signs do not set the 50 in Switzerland.** The 50 starts at 2.30.1 and ends at 2.53.1 (VRV art. 4a(2), SSV art. 22(1),(3)). The town sign stands where the loosely built area starts and must not stand after 2.30.1 (SSV art. 50(4)). So on the way in, the town sign comes before or together with 2.30.1. The two signs are often on the same post, but they need not be. Certainty high (text of the law).
- The 2.30.1 / 2.53.1 signs may be missing on unimportant side roads (SSV art. 22(4)).
- No town signs on Autobahnen or Autostrassen (SSV art. 50(1)).
- Inside towns, "innerorts" for other rules runs from 4.27/4.29 to 4.28/4.30 (SSV art. 1(4)); that is a different notion from the 50 area.

How a camera can tell Autobahn from Autostrasse:

- The symbol on the green sign: motorway with overbridge (4.01) vs. car front (4.03). Colour alone does not help: both are green, and direction signs to both are green (SSV art. 51(1)). Certainty high.
- The posted number: Autostrassen have the 100 signed (SSV art. 22(5)); Autobahnen usually carry no general-limit sign.
- Layout is not decisive: an Autostrasse may have separated carriageways too.

Where the signs stand:

- 4.01 and 4.03 stand at the start of the entry ramp. 4.02 and 4.04 stand on the exit ramp, shortly before the ordinary network (SSV art. 85(1)). So on entry the motorway sign is seen on the ramp, before any ramp limit sign; on exit, the end sign comes at the end of the ramp.
- 4.01 / 4.03 are also placed where an Autostrasse becomes an Autobahn or the other way round, but not on links between two motorways (SSV art. 85(2)).
- Signs stand on the right; they may be repeated on the left or hung over the carriageway (SSV art. 103).
- Prescription signs stand where the rule begins; a distance panel 5.01 announces a rule that starts later (SSV art. 16(3)).

## How long a speed sign holds

The rule (SSV art. 16(2)):

> "Die Signale «Höchstgeschwindigkeit» (2.30) … gelten bis zu den entsprechenden Ende-Signalen (2.53, … 2.58), höchstens aber bis zum Ende der nächsten Verzweigung."

So a 2.30 sign holds until its end sign, but at most until the end of the next junction; to continue, it must be repeated after the junction.

- **2.30.1 "50 generell" is the exception:** it holds in the whole densely built area of the town (SSV art. 16(2) last sentence), through junctions.
- **What a junction (Verzweigung) is:** crossings, forks and joins of carriageways. Exits from garages, car parks, factories, farms and the like, and cycle or field paths, are not junctions (VRV art. 1(8)).
- **Motorway entries and exits (Anschlüsse) are not junctions.** An Anschluss is where entry and exit ramps meet the carriageway of an Autobahn or Autostrasse (SSV art. 86(1)). BGE 128 IV 30 (4 Dec 2001) held that on motorways a Verzweigung exists only where motorways fork or join, not at Anschlüsse. A limit passes through an Anschluss. Certainty high.
- **Interchanges between motorways (Verzweigungen) are junctions, and the junction covers the whole interchange.** In BGE 128 IV 30 (A1 at Härkingen, where the A2 joins), the court held that the limit did not end at the fork of the carriageways; it held through the whole interchange until the end signs placed after the re-merge, at the end of the acceleration lane. Regeste: "Die signalisierte Höchstgeschwindigkeit gilt von der Stelle an, wo das Signal steht, bis zum Endesignal. Sie endet nicht schon nach der Gabelung der Fahrbahnen." Where the "end of the junction" lies in other layouts was not decided generally. Certainty high for Härkingen; medium for other layouts.
- **Ordinary junctions on ordinary roads end a 2.30 limit when it is not repeated.** BGer 6S.510/1999 (19 Jan 2000): a 60 sign inside a town, not repeated after a junction, no longer applied there. The court left open whether 50 generell then applied, but the driver, still inside the town, was bound by 60 anyway. Certainty high that the limit ends; what applies next is "the general limit of the road" (VRV art. 4a), certainty medium.
- **Signs act only in the direction they face** (6S.510/1999).
- **Repetition:** ASTRA 15016 §3.2 (p.14): on motorway traffic-control systems the limit is repeated after the end of the acceleration lane at Anschlüsse, Verzweigungen and rest areas; where that is impossible, the limit is shown on the entry ramp. On long sections, signs are repeated with panel 5.04 or given a length panel 5.03 (SSV art. 16(4)).
- **Ramps:** the motorway rules apply on the ramps between 4.01 and 4.02. BGE 128 II 131 judged speeding on a motorway exit ramp by the out-of-town scale (a sanction question, not the limit itself). Not found: a ruling on whether a ramp limit binds the carriageway after the merge. Ramp limits are signed on the ramp and, by art. 16(2), end at the latest at the end of the next junction; whether the merge into the carriageway counts as the end of a Verzweigung or of an Anschluss: the merge is part of the Anschluss (SSV art. 86(1)), so by BGE 128 IV 30 it is not a Verzweigung. In practice the carriageway's own limit or signs apply after the merge. Certainty medium (interpretation).
- **Signs placed unlawfully must still be obeyed** unless obviously void: BGer 6B_522/2012 (25 Jan 2013), https://servat.unibe.ch/dfr/bger/2013/130125_6B_522-2012.html. Certainty low: seen only in a search snippet, ruling not read.
- Not found: a Swiss ruling on traffic that enters after a sign (from a side road) without seeing it.

## End signs

| Sign | Meaning | Source |
|---|---|---|
| 2.53 "Ende der Höchstgeschwindigkeit" | Ends the signed limit; the general limit of the road applies. White disc, the number crossed by black diagonal stripes. | SSV art. 22(1), 32(1) |
| 2.53.1 "Ende der Höchstgeschwindigkeit 50 generell" | Ends the town 50; 80 applies (VRV art. 4a(3)). | SSV art. 22(3), 32(1) |
| 2.58 "Freie Fahrt" | Several restrictions for moving traffic end; the general rules apply again. Ends motorway roadworks; restrictions still in force must be repeated. White disc with one broad black diagonal band. | SSV art. 32(2) |
| 2.59.2 "Ende-Zonensignal", 2.59.4, 2.59.6 | End of a zone (30-zone; pedestrian zone; meeting zone); the general rules apply after it. | SSV art. 2a(3), annex 2 |
| 4.02 / 4.04 | End of Autobahn / Autostrasse: 80 applies, general rules again. | VRV art. 4a(3); SSV art. 45(1) |
| 4.01 / 4.03 | Not end signs, but they lift all earlier restrictions. | SSV art. 45(1) |

ASTRA 15016 p.21: in sections with a permanent truck overtaking ban, 2.53 is used rather than 2.58 (because 2.58 would also end the overtaking ban).

## Zones

- Zone signs only innerorts (SSV art. 2a(2)). A zone applies until its end sign; after it the general rules apply (art. 2a(3)). A new 30-zone or meeting-zone sign also ends the previous zone (art. 2a(3bis)).
- 30-zone, meeting zone (20) and pedestrian zone are allowed only on side roads that are not traffic-oriented (art. 2a(5)).
- Zone sign 2.59.1: white rectangle with "ZONE" above a 30 roundel (annex image). The zone limit holds through junctions inside the zone (art. 2a(3): "until the end sign"). Certainty high.

## Variable message signs

- **Legal status:** signs may be shown on Wechselsignalanlagen (variable sign systems) and over lanes on white rectangular panels (SSV art. 101(7)(a),(d)). Matrix signs may have a black background with a white symbol (SSV art. 16(1)), and illuminated signs may be on black rectangular panels (art. 101(7bis)). They have the same effect as fixed signs. Certainty high.
- **Lane-specific limits:** signs apply to the whole carriageway unless their placement over lanes shows they apply only to certain lanes (SSV art. 101(4)). ASTRA 15016 §3.2 (p.14): normally the same limit on all lanes; at Anschluss and Verzweigung areas the difference may be at most 20 km/h.
- **Steps:** reductions in 20 km/h steps; an increase may come in one step; cross-sections about 1000 m apart (400–1500 m) (ASTRA 15016 §3.2, p.14). Every cross-section should be able to show the lifting of the previous limit (same page).
- **Dark (blank) panel is a separate state from "end of limit":** ASTRA 15016 §3.4.1 (p.16) lists the states V100, V80, V80 durchgehend, V60, "Ende der Höchstgeschwindigkeit (V100, V80, V60, „Freie Fahrt")" and "dunkel (keine Anzeige)". ASTRA 15016 §3.8 (p.23) priority: V60 > V80 > V100 > end > dark. ASTRA 15019 annex IV.4 (p.88) gives "AUS / dunkel (hochprior)" 4900 and "AUS / dunkel (niedrigprior)" 0000, separate from 2.58 (0530) and 2.53 (0405–0415). These are control-logic priorities, not legal classes; a camera cannot tell "high" from "low" priority dark.
- **On a fault** the signs go to a defined base state (ASTRA 15016 §3.4, p.16). In the base state the controllable signs "sind ausgeschaltet bzw. zeigen die üblichen Signalbilder an" (ASTRA 15010 §2.1, p.8). So a dark panel says nothing reliable about the stretch.
- **Conclusion:** a dark panel shows no signal, so it is not 2.53 or 2.58 and does not end a limit. The last limit holds until an end sign, a new limit, a road-kind sign, or the end of the next Verzweigung (SSV art. 16(2)). Certainty medium-high: follows from art. 16(2) and 32 and the ASTRA state model; no ruling found on this exact sequence.

Scenarios (from the upload, checked):

| Seen on a Swiss motorway | Meaning | Safe action |
|---|---|---|
| Variable sign "80" readable | New signed limit | Take 80 after confirmation |
| "80", then a clearly dark panel, nothing else | A dark panel is not 2.53 | Do not set 120; do not treat dark as "end" |
| "80", then "100" | New explicit limit | Take 100 after confirmation |
| "80", then 2.53 | Explicit end | General limit of the road (120), if the road kind is known |
| "80", then 2.58 | Several restrictions end | Separate class; general limit of the road |
| "80", then an ordinary entry (Anschluss) | Not a Verzweigung (BGE 128 IV 30) | Keep 80 |
| "80", then an ordinary exit, staying on the motorway | Not a Verzweigung | Keep 80 |
| "100", then a motorway interchange (A1/A2) with no end sign | Junction covers the whole interchange; the limit does not end at the fork | Do not reset at the fork; at most after the interchange, if not repeated |
| No sign for many kilometres | Absence of a sign is not an end sign | Keep the limit; lower confidence, never raise |
| Panel dark because of glare, rain, occlusion, fault | Unreliable observation | UNKNOWN: no change |
| Static roadworks "60" and a dark electronic panel | Static sign binds | Keep 60 |

## Supplementary panels

- Panels are rectangular, white with black text; on matrix signs black with white text; they are as binding as signs (SSV art. 63(1),(3)).
- 5.01 Distanztafel: distance to where the rule starts (art. 64(1)). The sign then announces a limit that is not yet in force.
- 5.03 Streckenlänge: length of the section (art. 64(2)).
- 5.04 Wiederholungstafel: the sign is a repetition (art. 64(3)).
- A vehicle-class symbol limits the sign to that class (e.g. 5.22 Lastwagen, 5.26 Anhänger); "ausgenommen" or "gestattet" with a symbol exempts that class (art. 64(5)). A 60 with a truck panel does not apply to a car.
- 5.44 "Fahrzeuge mit einem Automatisierungssystem" exists in the panel list (annex 2). Its use with speed signs: not found.
- Not found: a wet-road panel in the SSV panel list. Time-of-day panels with speed signs: not verified.

## Special cases for a camera system

- **Truck-only limits:** a 2.30 with a truck or trailer panel must be ignored for a car (art. 64(5)).
- **Lane signs:** a number inside a lane-arrow sign (4.77.1, art. 59(3)) or on a gantry over one lane applies to that lane only (art. 101(4)).
- **Signs for the other road:** a sign facing a side road or a parallel road; signs act only in their facing direction (6S.510/1999).
- **Border signs 4.93:** information panels at the border list the general limits (50/80/100/120, SSV art. 61); they are not limit signs.
- **"50 generell" vs. plain 50:** 2.30.1 holds through junctions; a plain 2.30 "50" does not. The classifier should keep them apart if it can.
- **2.53 vs. 2.58:** both end the limit; 2.58 ends more. For speed both give the general limit of the road.
- **Liechtenstein** (LI) uses the same sign numbers but also has "30 generell" and "40 generell" (see li.md).

## For the comma

IMPLICIT_LIMITS row (correct in vision.py, certainty high):

```python
"CH": {"town": 50, "rural": 80, "expressway": 100, "motorway": 120},
```

Intersection rule: **partly** (SSV art. 16(2)). A 2.30 limit holds until its end sign, at most until the end of the next Verzweigung, unless repeated:

- ordinary roads: a crossing, fork or join of carriageways ends it; garage, car-park, farm and field-path exits do not (VRV art. 1(8));
- motorways and Autostrassen: Anschlüsse (entries/exits) do not end it; an interchange between motorways ends it only at the end of the whole interchange (BGE 128 IV 30);
- 2.30.1 "50 generell" and zone limits hold through junctions.

The code's MAP_SECONDS rule (the map's limit wins after a change of road) stands in for this. It is right in spirit, but note that a turn into a side road is the usual case; continuing straight through a junction ends the limit too, if no repeat sign stands after it.

Blank variable sign: **not an end sign.** Keep the last limit; never raise because of a dark panel (ASTRA 15016 §3.4.1, §3.8; SSV art. 16(2), 32).

Other rules the code should know:

1. **ROAD_SIGNS town mapping is imprecise in CH.** The 50 starts at 2.30.1 and ends at 2.53.1 (VRV art. 4a(2), SSV art. 22(3)), not at the town signs 4.27–4.30. Entering, the town sign stands before or with 2.30.1 (SSV art. 50(4)), so `town_entry → 50` errs low, which is safe. Leaving, `town_exit → 80` can be wrong when a lower 2.30 limit (e.g. 60) stands near the town end and continues past the town sign: the 60 holds until 2.53 or the next junction. Proposal: for CH and LI, the classes "50 generell" and "Ende 50 generell" should set 50 and 80; the town-exit plate should not raise the limit on its own. Certainty high in law, medium on how often it matters.
2. **4.01 and 4.03 lift all earlier limits** (SSV art. 45(1)). The code does this already (a road sign sets the general limit).
3. **4.02 / 4.04 give 80** even if the ordinary road beyond is in a town; the 50 then needs 2.30.1 (VRV art. 4a(3)). The code's `motorway_end → rural` matches.
4. **Entry ramps:** 4.01 stands at the start of the entry ramp (SSV art. 85(1)), so the RAMP_SECONDS logic (a number read soon after a motorway sign is the ramp's) fits Swiss placement.
5. **0 (end sign) gives the general limit of the known road kind** (2.53, 2.58). The code does this. When the road kind is unknown it shows 0 ("no limit"); for CH it would be safer to keep the last limit or use the map than show nothing. Certainty medium (design).
6. **No signed value above the general limit outside towns** (SSV art. 108(5)). A reading of 130 on a Swiss road is a misread or a foreign sign; a reading above 80 on a rural road, above 100 on an Autostrasse, above 120 on an Autobahn is not lawful. The code can use this as a plausibility check.
7. **Observation classes proposed in the upload** (sound): SPEED_LIMIT(n), END_SPEED_LIMIT (2.53), FREE_DRIVE (2.58), BLANK_VARIABLE_SIGN (panel seen, nothing lit), UNKNOWN (no panel, occlusion, glare). BLANK and UNKNOWN must never end or raise a limit.
8. **Safety rules (upload, sound):** never raise the set speed on an uncertain inference; a false 120 after an 80 is worse than a late raise; a single-frame "end" must not raise; absence of signs is not an end sign; separate "last number seen" from "legal limit inferred"; the driver remains responsible.
9. **Test cases (upload, checked against the law):**
   1. 80 → blank → blank: never END, no 120.
   2. 80 → real 2.53: end only with confidence and known road kind.
   3. 80 → 100 confirmed: 100.
   4. 80 → blank → 60: 60.
   5. 80 → ordinary entry → blank: keep 80 (Anschluss, BGE 128 IV 30).
   6. 80 → ordinary exit, staying on the motorway: keep 80.
   7. 100 → motorway interchange without 2.53: no reset at the fork (BGE 128 IV 30).
   8. 80 → 2.53 with static roadworks 60: never 120 against the 60.
   9. Panel hidden by a truck or glare: UNKNOWN, not BLANK, not END.
   10. One-frame false "end": no raise.
   11. Long stretch without signs: lower confidence; a timer running out is not "back to 120".
   12. Change of motorway or leaving the motorway network: recompute the general limit only on a confirmed road kind.
   13. Zone and "50 generell" signs: do not reuse plain 2.30 semantics (they hold through junctions).

## Sources

All checked 2026-10-29.

- VRV, SR 741.11, version 1 Oct 2026: https://www.fedlex.admin.ch/eli/cc/1962/1364_1409_1420/de (read from https://fedlex.data.admin.ch/filestore/fedlex.data.admin.ch/eli/cc/1962/1364_1409_1420/20261001/de/html/fedlex-data-admin-ch-eli-cc-1962-1364_1409_1420-20261001-de-html-1.html). Art. 1(3), 1(8), 4, 4a, 5.
- SSV, SR 741.21, version 1 Oct 2026: https://www.fedlex.admin.ch/eli/cc/1979/1961_1961_1961/de (read from https://fedlex.data.admin.ch/filestore/fedlex.data.admin.ch/eli/cc/1979/1961_1961_1961/20261001/de/html/fedlex-data-admin-ch-eli-cc-1979-1961_1961_1961-20261001-de-html-2.html). Art. 1(4),(6), 2a, 16, 22, 22a–c, 32, 44, 45, 50, 51, 59, 61, 63, 64, 84–87, 101, 103, 107, 108; annex 2.
- BGE 128 IV 30 (6S.579/2001, 4 Dec 2001): https://servat.unibe.ch/dfr/bge/c4128030.html (bger.ch blocks automated access).
- BGer 6S.510/1999 (19 Jan 2000): https://servat.unibe.ch/dfr/bger/2000/000119_6S-510-1999.html
- BGE 128 II 131: https://servat.unibe.ch/dfr/bge/c2128131.html
- BGer 6B_522/2012 (25 Jan 2013): https://servat.unibe.ch/dfr/bger/2013/130125_6B_522-2012.html (snippet only).
- ASTRA 15016 "Geschwindigkeitsharmonisierung und Gefahrenwarnung (GHGW)", Ausgabe 2015 V1.02, §3.2 p.14, §3.4/3.4.1 p.16, p.21, §3.8 p.23: https://www.astra.admin.ch/dam/de/sd-web/xfoQCurpGhav/astra_15016_geschwindigkeitsharmonisierungundgefahrenwarnungghgw.pdf
- ASTRA 15019 "Verkehrstechnische Regelungslogik", Ausgabe 2018 V1.04, annex IV.4 p.88: https://www.astra.admin.ch/dam/de/sd-web/J25yRpGv-xQQ/astra%2015019%20verkehrstechnische%20regelungslogik.pdf
- ASTRA 15010 "Betriebszustände Verkehrssteuerung", Ausgabe 2015 V1.02, §2.1 p.8: https://www.astra.admin.ch/dam/de/sd-web/LZaTWBZnn45x/astra_15010_betriebszustaende-verkehrssteuerung2015100.pdf

## Open points

- Legal effect of the exact sequence "80 shown → next panel dark" in a coordinated system: no ruling found. The answer above is an interpretation (medium-high).
- Where the "end of the next Verzweigung" lies on interchanges other than Härkingen, and whether a ramp merge ever counts as one: not decided in general.
- What applies after a 2.30 limit ends at a junction without repetition inside a town (50 generell or nothing signed): left open in 6S.510/1999.
- Traffic entering after a sign from a side road: no Swiss ruling found.
- Novice-driver rules outside the VRV (driving-licence ordinance): not searched.
- Time-of-day panels with speed signs and use of panel 5.44: not verified.
- Whether 6B_522/2012 says what the snippet says: not read.
