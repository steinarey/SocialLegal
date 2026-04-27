-- Pre-fill the municipalities table with the canonical list of Icelandic
-- municipalities. Idempotent: safe to run on an empty DB or one that already
-- has some entries. Existing rows are not modified.
--
-- Run from the host with:
--   docker compose exec -T db psql -U legal -d legal < seed_municipalities.sql

BEGIN;

INSERT INTO municipalities (id, name) VALUES
    (1,  'Reykjavíkurborg'),
    (2,  'Kópavogsbær'),
    (3,  'Hafnarfjarðarkaupstaður'),
    (4,  'Reykjanesbær'),
    (5,  'Garðabær'),
    (6,  'Akureyrarbær'),
    (7,  'Mosfellsbær'),
    (8,  'Sveitarfélagið Árborg'),
    (9,  'Akraneskaupstaður'),
    (10, 'Múlaþing'),
    (11, 'Fjarðabyggð'),
    (12, 'Seltjarnarnesbær'),
    (13, 'Vestmannaeyjabær'),
    (14, 'Skagafjörður'),
    (15, 'Suðurnesjabær'),
    (16, 'Borgarbyggð'),
    (17, 'Ísafjarðarbær'),
    (18, 'Hveragerðisbær'),
    (19, 'Norðurþing'),
    (20, 'Sveitarfélagið Ölfus'),
    (21, 'Sveitarfélagið Hornafjörður'),
    (22, 'Rangárþing eystra'),
    (23, 'Fjallabyggð'),
    (24, 'Rangárþing ytra'),
    (25, 'Dalvíkurbyggð'),
    (26, 'Sveitarfélagið Vogar'),
    (27, 'Snæfellsbær'),
    (28, 'Þingeyjarsveit'),
    (29, 'Bláskógabyggð'),
    (30, 'Húnabyggð'),
    (31, 'Sveitarfélagið Stykkishólmur'),
    (32, 'Vesturbyggð'),
    (33, 'Eyjafjarðarsveit'),
    (34, 'Húnaþing vestra'),
    (35, 'Mýrdalshreppur'),
    (36, 'Bolungarvíkurkaupstaður'),
    (37, 'Hörgársveit'),
    (38, 'Hrunamannahreppur'),
    (39, 'Grundarfjarðarbær'),
    (40, 'Grindavíkurbær'),
    (41, 'Hvalfjarðarsveit'),
    (42, 'Flóahreppur'),
    (43, 'Skeiða- og Gnúpverjahreppur'),
    (44, 'Grímsnes- og Grafningshreppur'),
    (45, 'Dalabyggð'),
    (46, 'Skaftárhreppur'),
    (47, 'Vopnafjarðarhreppur'),
    (48, 'Langanesbyggð'),
    (49, 'Svalbarðsstrandarhreppur'),
    (50, 'Sveitarfélagið Skagaströnd'),
    (51, 'Strandabyggð'),
    (52, 'Grýtubakkahreppur'),
    (53, 'Kjósarhreppur'),
    (54, 'Ásahreppur'),
    (55, 'Reykhólahreppur'),
    (56, 'Súðavíkurhreppur'),
    (57, 'Eyja- og Miklaholtshreppur'),
    (58, 'Kaldrananeshreppur'),
    (59, 'Fljótsdalshreppur'),
    (60, 'Skorradalshreppur'),
    (61, 'Árneshreppur'),
    (62, 'Tjörneshreppur')
ON CONFLICT (name) DO NOTHING;

-- Bump the SERIAL sequence past the highest manual id so subsequent inserts
-- via the API (which lets the SERIAL assign the id) do not collide with
-- these reserved values. Safe to re-run.
SELECT setval(
    pg_get_serial_sequence('municipalities', 'id'),
    GREATEST(COALESCE((SELECT MAX(id) FROM municipalities), 0), 62)
);

COMMIT;
