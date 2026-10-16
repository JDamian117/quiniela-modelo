"""Trazabilidad (spec §3.4): cada dato externo registra fuente, fecha de consulta y estado; los fallos NO se ocultan."""
from __future__ import annotations

from datetime import datetime, timezone

_LOG: list[dict] = []


def record(source: str, status: str, detail: str = "", n: int | None = None) -> None:
    """status: ok | fallo | neutro (se usó valor neutro por falta de dato) | aviso."""
    _LOG.append({"fuente": source, "estado": status, "detalle": detail, "n": n,
                 "consultado_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")})


def get() -> list[dict]:
    return list(_LOG)


def clear() -> None:
    _LOG.clear()


def failures() -> list[dict]:
    return [r for r in _LOG if r["estado"] in ("fallo", "neutro", "aviso")]


def summary() -> str:
    """Resumen legible: agrupa registros repetidos por familia (p. ej. 'Lesiones · <equipo>') para no ahogar los fallos reales."""
    ok = sum(r["estado"] == "ok" for r in _LOG)
    groups: dict[tuple, list[dict]] = {}
    for r in failures():
        groups.setdefault((r["estado"], r["fuente"].split(" · ")[0]), []).append(r)
    lines = [f"[fuentes] {ok} ok, {len(failures())} con fallo/aviso/valor neutro"]
    for (estado, fam), rs in groups.items():
        detail = rs[0]["detalle"] if len(rs) == 1 else f"{len(rs)} casos; ej.: {rs[0]['detalle']}"
        lines.append(f"  - {estado.upper():7s} {fam if len(rs) > 1 else rs[0]['fuente']}: {detail}")
    return "\n".join(lines)
