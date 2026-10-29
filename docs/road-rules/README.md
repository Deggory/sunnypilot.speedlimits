# Road rules by country

The speed rules the comma needs, read from each country's laws and checked on 29 Oct 2026: one file per country
(`<cc>.md`, ISO code), each with its sources, how sure each point is and what stays open. They decide the tables of the
comma's speed-limit memory (sunnypilot, branch `speed-limits`, `openpilot/sunnypilot/speed_limits/vision.py`).

Not legal advice: the files cite the texts read, and mark what was read from them (high), inferred (medium) or not
found (low). Case law was searched for some countries only.

## At a glance

General limits of a passenger car in km/h, where no number is signed. "Junction" is whether a speed sign ends at the
next intersection without an end sign. "Road sign" is whether a sign of the kind of road (motorway, expressway, town)
ends a limit held.

| Country | Town | Rural | Expressway | Motorway | Junction ends a sign | Road sign ends the limit | File |
|---|---|---|---|---|---|---|---|
| Switzerland | 50 | 80 | 100 | 120 | partly: yes on ordinary roads, not motorway entries/exits, not "50 generell" or zones | yes | [ch](ch.md) |
| Liechtenstein | 50 | 80 | 80 | 80 | partly, as CH | yes | [li](li.md) |
| Germany | 50 | 100 | 100 (single carriageway) | none | no | yes | [de](de.md) |
| Austria | 50 | 100 | 100 | 130 | no | yes | [at](at.md) |
| Italy | 50 | 90 | 110 | 130 | yes, not zones | yes | [it](it.md) |
| France | 50 | 80 | 110 (divided) | 130 | not verified | yes | [fr](fr.md) |
| Netherlands | 50 | 80 | 100 | 130 (100 by day on many) | no | yes | [nl](nl.md) |
| Belgium | 50 (Brussels 30) | 70 (Wallonia 90) | as rural | 120 | yes, not a limit at the town sign or zones | yes | [be](be.md) |
| Luxembourg | 50 | 90 | 90 | 130 | no (a turn ends it) | yes | [lu](lu.md) |
| Spain | 50 (30 one lane, 20 shared) | 90 | 90 | 120 | partly: leaving a minor road onto a priority road | yes | [es](es.md) |
| Portugal | 50 | 90 | 100 | 120 | partly: repeated after level junctions | yes | [pt](pt.md) |
| Ireland | 50 | 80 (local 60, national 100) | 100 | 120 | no (a turn ends it) | town nameplates: no | [ie](ie.md) |
| Great Britain | 30 mph (Wales 20) | 60 mph | 70 mph | 70 mph | no | town nameplates: no | [gb](gb.md) |
| Denmark | 50 | 80 | 80 | 130 | no | yes, unless repeated within 60 m | [dk](dk.md) |
| Sweden | 50 | 70 | 70 | 110 | no | **no** | [se](se.md) |
| Norway | 50 | 80 | 80 | 80 | no (a turn ends it) | **no** | [no](no.md) |
| Finland | 50 | 80 | 80 | 80 | no (a turn ends it) | **no** | [fi](fi.md) |
| Poland | 50 | 90 | 100 single, 120 dual | 140 | **yes**, not under D-42 / E-17a or zones | yes | [pl](pl.md) |
| Czechia | 50 | 90 | 110 | 130 | **yes**, interchanges too (medium) | yes | [cz](cz.md) |
| Slovakia | 50 | 90 | 90 | 130 | partly: roadside signs, until 2029 | yes | [sk](sk.md) |
| Hungary | 50 | 90 | 110 | 130 | **yes** | yes | [hu](hu.md) |
| Slovenia | 50 | 90 | 110 | 130 | **yes**, not the ramp merge | yes | [si](si.md) |
| Croatia | 50 | 90 | 110 | 130 | partly: the sign is repeated after a junction | yes | [hr](hr.md) |

A dark or blank variable panel is not an end sign anywhere it was found: the limit held stays (CH, DE, DK, ES, NL,
PT; in LU it probably ends a limit set by the panel). Nowhere does a dark panel raise the limit.

## In the comma's code

- `IMPLICIT_LIMITS`: the table above, in km/h, the lower value where it depends on what the camera cannot tell
  (single or dual carriageway, the region of Belgium). GB is left out: its signs are in mph and the classifier reads
  km/h. `None` means no general limit (the German motorway): never use it for a country that has one.
- `ROAD_SIGN_KEEPS_LIMIT`: SE, NO and FI, where a motorway, expressway or town sign only says which road the car is on.
- The country is the one chosen for the maps on the comma (Settings › OSM, `OsmLocationName`), else CH.
- Junctions: the vision side never reads the map; map and vision meet only in sunnypilot's resolver ("Map Data +
  Vision"). The rule of the countries where a sign ends at the next junction (CH, LI, IT, BE, PL, CZ, HU, SI and partly
  SK, HR, PT, ES) waits for the road-type network to see the turn from the camera and the car's own signals (yaw,
  steering, blinkers); it ends a number read from a sign, never a zone or a town-wide limit.

## Not handled yet

- Limits with a condition: rain (FR, IT, LU), time windows (NL), lanes (ES), a town around a motorway (CZ, ES),
  regions (BE, GB-Wales).
- Zone signs and town-wide limits as their own kind, so a junction does not end them.
- The country from the GPS instead of the map setting, for a drive across a border.
