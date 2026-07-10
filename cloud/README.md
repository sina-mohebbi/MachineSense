# cloud/ — Phase 3: EMQX + TimescaleDB + Grafana

Self-hosted, $0, no credit card. Brings up the whole backend with one command.

```bash
docker compose up -d
```

| Service | URL | Default login |
|---|---|---|
| EMQX dashboard | http://localhost:18083 | admin / public |
| Grafana | http://localhost:3000 | admin / admin |
| TimescaleDB | localhost:5432 | machinesense / machinesense |

## Why EMQX (vs the Mosquitto stack in library-desk-sense)

EMQX has a built-in **rule engine + data bridges**, so it filters telemetry and writes
straight to TimescaleDB by itself — **replacing the custom Python proxy** from the last
project. It also gives MQTT 5.0, auth/ACL, TLS, a live monitoring dashboard, and scales
to a real device fleet.

## Data flow (configured in Phase 3)

1. ESP32 publishes `machinesense/<machine>/<id>/telemetry`
   `{ "score": 0.83, "anomaly": true, "latency_ms": 6, "ts": ... }`
2. EMQX **rule**: `SELECT ... FROM "machinesense/#"` → **TimescaleDB data bridge** (INSERT).
3. Grafana reads TimescaleDB (SQL) → anomaly-score timeline + threshold line + alerts.
4. Config push: retained message on `machinesense/<machine>/<id>/config` carries the
   anomaly threshold; the device subscribes on boot (lightweight "device shadow").

## TODO (Phase 3)

- [ ] `timescaledb/init.sql` — hypertable for telemetry
- [ ] EMQX rule + TimescaleDB bridge (via dashboard or `emqx.conf`)
- [ ] `grafana/provisioning/` — datasource + anomaly dashboard
- [ ] TLS certs + MQTT auth
