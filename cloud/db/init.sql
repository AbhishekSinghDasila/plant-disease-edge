-- Runs once, when the postgres volume is first created.
-- Every MQTT message is kept in telemetry (raw JSON); the other tables are typed views of the important ones.
CREATE TABLE IF NOT EXISTS telemetry (
    id      BIGSERIAL PRIMARY KEY,
    ts      TIMESTAMPTZ NOT NULL DEFAULT now(),
    topic   TEXT NOT NULL,
    kind    TEXT NOT NULL,          -- last topic part: climate, soil, diagnosis, ...
    device  TEXT,
    data    JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS telemetry_kind_ts ON telemetry (kind, ts DESC);

CREATE TABLE IF NOT EXISTS images (
    image_id   TEXT PRIMARY KEY,
    ts         TIMESTAMPTZ NOT NULL DEFAULT now(),
    device     TEXT,
    path       TEXT NOT NULL,
    meta       JSONB
);

CREATE TABLE IF NOT EXISTS diagnoses (
    id         BIGSERIAL PRIMARY KEY,
    ts         TIMESTAMPTZ NOT NULL DEFAULT now(),
    source     TEXT NOT NULL,       -- edge | cloud
    device     TEXT,
    image_id   TEXT,
    class_id   TEXT,                -- crop.disease (best guess)
    crop       TEXT,
    status     TEXT,                -- healthy | disease | not_sure
    confidence REAL,
    data       JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS diagnoses_ts ON diagnoses (ts DESC);
CREATE INDEX IF NOT EXISTS diagnoses_image ON diagnoses (image_id);

CREATE TABLE IF NOT EXISTS advice (
    id         BIGSERIAL PRIMARY KEY,
    ts         TIMESTAMPTZ NOT NULL DEFAULT now(),
    image_id   TEXT,
    class_id   TEXT,
    verdict    TEXT,
    data       JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS irrigation_events (
    id         BIGSERIAL PRIMARY KEY,
    ts         TIMESTAMPTZ NOT NULL DEFAULT now(),
    need_water BOOLEAN,
    urgency    TEXT,
    data       JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    id         BIGSERIAL PRIMARY KEY,
    ts         TIMESTAMPTZ NOT NULL DEFAULT now(),
    level      TEXT,                -- info | warning | critical
    kind       TEXT,                -- disease | water | offline | sensor
    message    TEXT,
    data       JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS alerts_ts ON alerts (ts DESC);
