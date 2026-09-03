"""Record and watchlist cross-check against a seeded local table.

**This is simulated data and nothing here touches a real system.** No student
project has access to Interpol SLTD, a national lookout circular, or any state
BOLO list, and implying otherwise in a demo is the kind of claim that collapses
under one question from a judge. Every result carries a `source` string saying so,
and the officer dashboard renders it.

The production integration point is named rather than faked: swap
`SimulatedRecordStore` for a client against the issuing authority's API under
MHA's data-governance terms, keeping this interface.

Records are held in memory. Persisting them to Postgres buys nothing for a
prototype whose entire record set is a handful of rows, and it would make the
demo depend on a database being up.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.schemas import DBCrosscheckResult

SOURCE_LABEL = "simulated local record set (no live watchlist access)"


@dataclass(frozen=True)
class DocumentRecord:
    """One row in the simulated issuing-authority record set."""

    document_number: str
    surname: str
    given_names: str
    nationality: str
    status: str                       # active | expired | reported_lost | revoked
    blacklisted: bool = False
    blacklist_reason: str | None = None
    prior_crossings: tuple[str, ...] = field(default_factory=tuple)


# Seeded demo records. The names are fictional and match the specimen documents
# generated in ml/data_prep -- no real identity appears anywhere in this project.
SEED_RECORDS: tuple[DocumentRecord, ...] = (
    DocumentRecord(
        document_number="Z9081234",
        surname="SHARMA",
        given_names="ANANYA DEVI",
        nationality="IND",
        status="active",
        prior_crossings=("ICP Attari, 11 Feb 2026 — cleared",),
    ),
    DocumentRecord(
        document_number="M7712854",
        surname="VENKATESH",
        given_names="PRIYA",
        nationality="IND",
        status="active",
    ),
    DocumentRecord(
        document_number="K4420917",
        surname="RATHORE",
        given_names="DEVENDRA SINGH",
        nationality="IND",
        status="reported_lost",
        blacklisted=True,
        blacklist_reason="document reported lost by the holder and since circulated",
    ),
    DocumentRecord(
        document_number="R1180563",
        surname="IYER",
        given_names="MEENAKSHI",
        nationality="IND",
        status="revoked",
        blacklisted=True,
        blacklist_reason="issuing authority revoked this document",
    ),
)


class SimulatedRecordStore:
    """Lookup over the seeded records."""

    def __init__(self, records: tuple[DocumentRecord, ...] = SEED_RECORDS) -> None:
        self._by_number = {r.document_number.upper(): r for r in records}

    def find(self, document_number: str | None) -> DocumentRecord | None:
        if not document_number:
            return None
        return self._by_number.get(document_number.replace(" ", "").upper())

    def __len__(self) -> int:
        return len(self._by_number)


_store = SimulatedRecordStore()


def cross_check(document_number: str | None) -> DBCrosscheckResult:
    """Look up a document number and describe what was found."""
    if not document_number:
        return DBCrosscheckResult(
            performed=False,
            source=SOURCE_LABEL,
            detail="No document number was extracted, so no record lookup was possible.",
        )

    record = _store.find(document_number)

    if record is None:
        return DBCrosscheckResult(
            performed=True,
            found=False,
            blacklisted=False,
            source=SOURCE_LABEL,
            detail=(
                f"Document {document_number} has no matching entry in the "
                f"{len(_store)}-record simulated set."
            ),
        )

    if record.blacklisted:
        return DBCrosscheckResult(
            performed=True,
            found=True,
            blacklisted=True,
            status=record.status,
            source=SOURCE_LABEL,
            detail=(
                f"Document {document_number} matches a flagged record: "
                f"{record.blacklist_reason}. Status recorded as {record.status}."
            ),
        )

    history = (
        f" Prior crossings on record: {'; '.join(record.prior_crossings)}."
        if record.prior_crossings
        else " No prior crossings on record."
    )
    return DBCrosscheckResult(
        performed=True,
        found=True,
        blacklisted=False,
        status=record.status,
        source=SOURCE_LABEL,
        detail=(
            f"Document {document_number} matches an active record for "
            f"{record.given_names} {record.surname} ({record.nationality}), status "
            f"{record.status}, with no flag against it.{history}"
        ),
    )
