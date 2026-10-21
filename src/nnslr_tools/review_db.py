"""Review database of the web review: one SQLite file under the data root.

Standard library only. The review server writes here and nowhere else; the
pre-annotation reports stay read-only. A confirmed frame is stored only after it
passed the same checks as ``nnslr import-annotations`` (the shared functions in
:mod:`nnslr_tools.annotations`), and every save is also appended to a history
table that cannot be updated or deleted.

``nnslr import-annotations --from-db`` turns the confirmed frames back into the
review JSONL lines the page export would write and imports them through the
normal import path.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from nnslr_tools.annotations import (
    DatasetError,
    import_review_sources,
    jsonl_bytes,
    load_jsonl,
    require,
    reviewed_frame_annotations,
    validate_dataset,
)
from nnslr_tools.manifest import parse_route_id

SCHEMA_VERSION = 1
DB_RELATIVE_PATH = Path('annotations') / 'review.sqlite'

# Fields the reviewer can change on a box. Everything else on a proposal (score,
# model label, class, evidence) comes from the read-only report on the server.
EDITABLE_DETECTION_KEYS = (
    'bbox_xyxy', 'family', 'value_kph', 'applicability', 'supplementary_panel',
    'visibility', 'occlusion', 'temporary', 'variable_display', 'notes',
)

# The annotation id only depends on frame and box identity, never on this hash,
# so a placeholder is enough to find which reviewed frames are already imported.
PLACEHOLDER_SOURCE_HASH = '0' * 64

SCHEMA_STATEMENTS = (
    """CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER NOT NULL,
        applied_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS frame_reviews (
        route_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        frame_key TEXT NOT NULL,
        segment_index INTEGER NOT NULL,
        output_index INTEGER NOT NULL,
        payload TEXT NOT NULL,
        reviewer TEXT NOT NULL,
        confirmed INTEGER NOT NULL CHECK (confirmed IN (0, 1)),
        revision INTEGER NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (route_id, run_id, frame_key)
    )""",
    """CREATE TABLE IF NOT EXISTS frame_review_history (
        history_id INTEGER PRIMARY KEY AUTOINCREMENT,
        route_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        frame_key TEXT NOT NULL,
        revision INTEGER NOT NULL,
        payload TEXT NOT NULL,
        reviewer TEXT NOT NULL,
        confirmed INTEGER NOT NULL CHECK (confirmed IN (0, 1)),
        saved_at TEXT NOT NULL,
        client TEXT NOT NULL
    )""",
    """CREATE INDEX IF NOT EXISTS frame_review_history_frame
        ON frame_review_history (route_id, run_id, frame_key)""",
    """CREATE TRIGGER IF NOT EXISTS frame_review_history_no_update
        BEFORE UPDATE ON frame_review_history
        BEGIN SELECT RAISE(ABORT, 'frame_review_history is append-only'); END""",
    """CREATE TRIGGER IF NOT EXISTS frame_review_history_no_delete
        BEFORE DELETE ON frame_review_history
        BEGIN SELECT RAISE(ABORT, 'frame_review_history is append-only'); END""",
)

# Italian text shown by the review page for each import-annotations reason code.
REASON_MESSAGES = {
    'invalid_json': 'Dati JSON non validi.',
    'invalid_schema': 'Struttura della revisione non valida.',
    'unknown_enum': 'Valore non ammesso in uno dei campi a scelta.',
    'unreviewed_proposal': 'Il fotogramma non risulta confermato.',
    'missing_provenance': 'Mancano i dati di provenienza della revisione.',
    'invalid_review_timestamp': 'Data della revisione non valida.',
    'invalid_identity': 'Identità del fotogramma o di un riquadro non valida.',
    'invalid_route_id': 'Identificativo della route non valido.',
    'invalid_dimensions': 'Dimensioni del fotogramma non valide.',
    'invalid_group': 'Sito fisico e incontro devono essere vuoti oppure un identificativo (non “unknown”).',
    'capture_time_inconsistency': 'Dati temporali del fotogramma incoerenti.',
    'illegal_speed': 'Il valore deve essere un numero intero tra 5 e 250 km/h.',
    'value_family_mismatch': 'Questo tipo di cartello non può avere un valore in km/h.',
    'value_state_inconsistency': 'Valore in km/h incoerente con il tipo scelto.',
    'invalid_bbox': 'Riquadro non valido: deve stare dentro l’immagine e avere larghezza e altezza.',
    'invalid_score': 'Punteggio della proposta non valido.',
    'missing_image': 'L’immagine originale del fotogramma non si trova sul server.',
    'unsafe_image_path': 'Percorso dell’immagine non consentito.',
    'absolute_path': 'Percorso dell’immagine non consentito.',
    'remote_path': 'Percorso dell’immagine non consentito.',
    'path_traversal': 'Percorso dell’immagine non consentito.',
    'hash_mismatch': 'L’immagine originale è cambiata dopo l’estrazione: rigenera il report.',
    'invalid_image': 'L’immagine originale è danneggiata.',
    'image_dimensions_mismatch': 'Le dimensioni dell’immagine non corrispondono al report.',
    'duplicate_identity': 'Lo stesso riquadro compare due volte nel fotogramma.',
    'contradictory_frame_labels': 'Il fotogramma risulta sia vuoto sia con riquadri.',
    'empty_dataset': 'Nessuna annotazione da salvare.',
}


class ReviewRejected(ValueError):
    """A save the server refuses; ``message`` is the Italian text for the page."""

    def __init__(self, status: int, reason: str, message: str):
        self.status = status
        self.reason = reason
        self.message = message
        super().__init__(f'{reason}: {message}')


class ReviewStoreError(ValueError):
    pass


def utc_now() -> str:
    """UTC time in the same form as the page's Date.toISOString()."""
    text = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
    return text.replace('+00:00', 'Z')


def db_path(root: Path) -> Path:
    return Path(root) / DB_RELATIVE_PATH


def italian_message(reason: str, detail: str = '') -> str:
    if reason == 'missing_provenance' and 'reviewer' in detail:
        text = 'Indica un nome o pseudonimo del revisore.'
    elif reason in REASON_MESSAGES:
        text = REASON_MESSAGES[reason]
    else:
        text = 'Revisione non accettata.'
    if detail:
        return f'{text} (motivo: {reason}; {detail})'
    return f'{text} (motivo: {reason})'


class ReviewStore:
    """Current review per (route, run, frame) plus an append-only history.

    Every call opens its own SQLite connection, so the store can be shared by
    the threads of the HTTP server. Writes are serialized by a lock and run in
    an IMMEDIATE transaction; the file uses WAL so reads never wait for writes.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._write_lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA busy_timeout = 30000')
        return connection

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._write_lock:
            connection = self._connect()
            try:
                connection.execute('PRAGMA journal_mode = WAL')
                connection.execute('BEGIN IMMEDIATE')
                try:
                    for statement in SCHEMA_STATEMENTS:
                        connection.execute(statement)
                    version = connection.execute('SELECT MAX(version) FROM schema_version').fetchone()[0]
                    if version is None:
                        connection.execute('INSERT INTO schema_version (version, applied_at) VALUES (?, ?)',
                                           (SCHEMA_VERSION, utc_now()))
                    elif version > SCHEMA_VERSION:
                        raise ReviewStoreError(f'review database {self.path} has schema version {version}; '
                                               f'this nnslr understands up to {SCHEMA_VERSION}')
                    connection.execute('COMMIT')
                except BaseException:
                    connection.execute('ROLLBACK')
                    raise
            finally:
                connection.close()

    def schema_version(self) -> int:
        connection = self._connect()
        try:
            return connection.execute('SELECT MAX(version) FROM schema_version').fetchone()[0]
        finally:
            connection.close()

    def save_review(self, *, route_id: str, run_id: str, frame_key: str, segment_index: int,
                    output_index: int, payload: dict, reviewer: str, client: str,
                    base_revision: int | None = None) -> dict:
        """Store a confirmed frame. The payload must already be validated."""
        text = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
        now = utc_now()
        with self._write_lock:
            connection = self._connect()
            try:
                connection.execute('BEGIN IMMEDIATE')
                try:
                    current = _current_revision(connection, route_id, run_id, frame_key)
                    _check_base_revision(base_revision, current)
                    revision = current + 1
                    if current == 0:
                        connection.execute(
                            'INSERT INTO frame_reviews (route_id, run_id, frame_key, segment_index, output_index, '
                            'payload, reviewer, confirmed, revision, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)',
                            (route_id, run_id, frame_key, segment_index, output_index, text, reviewer, revision, now))
                    else:
                        connection.execute(
                            'UPDATE frame_reviews SET payload = ?, reviewer = ?, confirmed = 1, revision = ?, '
                            'updated_at = ? WHERE route_id = ? AND run_id = ? AND frame_key = ?',
                            (text, reviewer, revision, now, route_id, run_id, frame_key))
                    _append_history(connection, route_id, run_id, frame_key, revision, text, reviewer, 1, now, client)
                    connection.execute('COMMIT')
                except BaseException:
                    connection.execute('ROLLBACK')
                    raise
            finally:
                connection.close()
        return self.review(route_id, run_id, frame_key)

    def withdraw_review(self, *, route_id: str, run_id: str, frame_key: str, reviewer: str, client: str,
                        base_revision: int | None = None) -> dict:
        """Mark a saved frame as no longer confirmed; its last payload is kept."""
        now = utc_now()
        with self._write_lock:
            connection = self._connect()
            try:
                connection.execute('BEGIN IMMEDIATE')
                try:
                    row = connection.execute(
                        'SELECT payload, revision FROM frame_reviews WHERE route_id = ? AND run_id = ? AND frame_key = ?',
                        (route_id, run_id, frame_key)).fetchone()
                    if row is None:
                        raise ReviewRejected(404, 'not_found',
                                             'Nessuna revisione salvata da annullare per questo fotogramma.')
                    _check_base_revision(base_revision, row['revision'])
                    revision = row['revision'] + 1
                    connection.execute(
                        'UPDATE frame_reviews SET reviewer = ?, confirmed = 0, revision = ?, updated_at = ? '
                        'WHERE route_id = ? AND run_id = ? AND frame_key = ?',
                        (reviewer, revision, now, route_id, run_id, frame_key))
                    _append_history(connection, route_id, run_id, frame_key, revision, row['payload'], reviewer, 0,
                                    now, client)
                    connection.execute('COMMIT')
                except BaseException:
                    connection.execute('ROLLBACK')
                    raise
            finally:
                connection.close()
        return self.review(route_id, run_id, frame_key)

    def review(self, route_id: str, run_id: str, frame_key: str) -> dict | None:
        connection = self._connect()
        try:
            row = connection.execute(
                'SELECT * FROM frame_reviews WHERE route_id = ? AND run_id = ? AND frame_key = ?',
                (route_id, run_id, frame_key)).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return _review_from_row(row)

    def reviews(self, route_id: str, run_id: str, *, confirmed_only: bool = False) -> list[dict]:
        query = 'SELECT * FROM frame_reviews WHERE route_id = ? AND run_id = ?'
        if confirmed_only:
            query = query + ' AND confirmed = 1'
        query = query + ' ORDER BY segment_index, output_index'
        connection = self._connect()
        try:
            rows = connection.execute(query, (route_id, run_id)).fetchall()
        finally:
            connection.close()
        result = []
        for row in rows:
            result.append(_review_from_row(row))
        return result

    def confirmed_count(self, route_id: str, run_id: str) -> int:
        connection = self._connect()
        try:
            return connection.execute(
                'SELECT COUNT(*) FROM frame_reviews WHERE route_id = ? AND run_id = ? AND confirmed = 1',
                (route_id, run_id)).fetchone()[0]
        finally:
            connection.close()

    def confirmed_count_in_other_runs(self, route_id: str, run_id: str) -> int:
        connection = self._connect()
        try:
            return connection.execute(
                'SELECT COUNT(*) FROM frame_reviews WHERE route_id = ? AND run_id <> ? AND confirmed = 1',
                (route_id, run_id)).fetchone()[0]
        finally:
            connection.close()

    def route_ids(self) -> list[str]:
        connection = self._connect()
        try:
            rows = connection.execute('SELECT DISTINCT route_id FROM frame_reviews ORDER BY route_id').fetchall()
        finally:
            connection.close()
        result = []
        for row in rows:
            result.append(row[0])
        return result

    def history(self, route_id: str, run_id: str, frame_key: str) -> list[dict]:
        connection = self._connect()
        try:
            rows = connection.execute(
                'SELECT * FROM frame_review_history WHERE route_id = ? AND run_id = ? AND frame_key = ? '
                'ORDER BY history_id', (route_id, run_id, frame_key)).fetchall()
        finally:
            connection.close()
        result = []
        for row in rows:
            entry = dict(row)
            entry['payload'] = json.loads(entry['payload'])
            entry['confirmed'] = entry['confirmed'] == 1
            result.append(entry)
        return result


def _current_revision(connection: sqlite3.Connection, route_id: str, run_id: str, frame_key: str) -> int:
    row = connection.execute(
        'SELECT revision FROM frame_reviews WHERE route_id = ? AND run_id = ? AND frame_key = ?',
        (route_id, run_id, frame_key)).fetchone()
    if row is None:
        return 0
    return row[0]


def _check_base_revision(base_revision: int | None, current: int) -> None:
    if base_revision is None:
        return
    if base_revision != current:
        raise ReviewRejected(409, 'revision_conflict',
                             'Questo fotogramma è stato salvato da un’altra pagina nel frattempo: '
                             'ricarica la pagina per vedere la versione attuale.')


def _append_history(connection: sqlite3.Connection, route_id: str, run_id: str, frame_key: str, revision: int,
                    payload: str, reviewer: str, confirmed: int, saved_at: str, client: str) -> None:
    connection.execute(
        'INSERT INTO frame_review_history (route_id, run_id, frame_key, revision, payload, reviewer, confirmed, '
        'saved_at, client) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
        (route_id, run_id, frame_key, revision, payload, reviewer, confirmed, saved_at, client))


def _review_from_row(row: sqlite3.Row) -> dict:
    return {
        'route_id': row['route_id'],
        'run_id': row['run_id'],
        'frame_key': row['frame_key'],
        'segment_index': row['segment_index'],
        'output_index': row['output_index'],
        'payload': json.loads(row['payload']),
        'reviewer': row['reviewer'],
        'confirmed': row['confirmed'] == 1,
        'revision': row['revision'],
        'updated_at': row['updated_at'],
    }


def public_review(review: dict) -> dict:
    """What the page needs to restore one saved frame."""
    payload = review['payload']
    return {
        'route_id': review['route_id'],
        'run_id': review['run_id'],
        'frame_key': review['frame_key'],
        'confirmed': review['confirmed'],
        'revision': review['revision'],
        'reviewer': review['reviewer'],
        'updated_at': review['updated_at'],
        'review_timestamp': payload.get('review_timestamp'),
        'detections': payload.get('detections', []),
        'site_group': payload.get('site_group'),
        'encounter_id': payload.get('encounter_id'),
    }


def _reviewed_detection(item, originals: dict):
    """One reviewed box: the original proposal (or a manual box) plus the reviewer's edits."""
    if not isinstance(item, dict):
        # Left as it is: the import checks reject it with their own reason.
        return item
    detection_id = item.get('detection_id')
    if isinstance(detection_id, str) and detection_id in originals:
        detection = copy.deepcopy(originals[detection_id])
    else:
        detection = {'score': None, 'model_label': None, 'class_id': None}
    for key in EDITABLE_DETECTION_KEYS:
        if key in item:
            detection[key] = copy.deepcopy(item[key])
    if 'detection_id' in item:
        detection['detection_id'] = detection_id
    detection['review_state'] = 'reviewed'
    if detection.get('value_kph') is None:
        detection['value_state'] = 'unknown'
    else:
        detection['value_state'] = 'reviewed'
    return detection


def build_reviewed_frame(report_row: dict, body: dict, timestamp: str) -> dict:
    """The reviewed frame line, the same as the page's "Scarica revisioni JSONL" writes it.

    Frame identity, image, hashes and the original proposals come from the
    report on the server; the request only brings the reviewer's decisions.
    """
    originals = {}
    proposals = report_row.get('detections')
    if isinstance(proposals, list):
        for proposal in proposals:
            if isinstance(proposal, dict) and isinstance(proposal.get('detection_id'), str):
                originals[proposal['detection_id']] = proposal
    detections = []
    for item in body['detections']:
        detections.append(_reviewed_detection(item, originals))
    reviewer = body.get('reviewer')
    if isinstance(reviewer, str):
        reviewer = reviewer.strip()
    site_group = body.get('site_group')
    if site_group is None:
        site_group = report_row.get('site_group')
    encounter_id = body.get('encounter_id')
    if encounter_id is None:
        encounter_id = report_row.get('encounter_id')
    frame = copy.deepcopy(report_row)
    frame['reviewer'] = reviewer
    frame['review_timestamp'] = timestamp
    frame['site_group'] = site_group
    frame['encounter_id'] = encounter_id
    frame['kind'] = 'reviewed_frame'
    frame['ground_truth'] = False
    frame['training_ready'] = False
    frame['review_state'] = 'reviewed'
    frame['label_basis'] = 'human_review'
    frame['source_proposals'] = copy.deepcopy(proposals)
    frame['detections'] = detections
    return frame


def validate_reviewed_frame(frame: dict, root: Path) -> None:
    """Apply the import-annotations rules to one reviewed frame; raise ReviewRejected(400)."""
    try:
        source_hash = hashlib.sha256(jsonl_bytes([frame])).hexdigest()
        rows = reviewed_frame_annotations(frame, source_hash, dataset_kind='gold', name='web review')
        report = validate_dataset(rows, root, dataset_kind='gold')
    except (ValueError, OSError) as exc:
        reason = getattr(exc, 'reason', 'invalid_schema')
        detail = getattr(exc, 'detail', str(exc))
        raise ReviewRejected(400, reason, italian_message(reason, detail)) from exc
    if not report['valid']:
        error = report['errors'][0]
        raise ReviewRejected(400, error['reason'], italian_message(error['reason'], error.get('detail', '')))


def save_frame_review(store: ReviewStore, root: Path, *, route_id: str, run_id: str, report_row: dict,
                      body, client: str) -> dict:
    """Handle one POST of the page: confirm (validated, stored) or withdraw a frame."""
    if not isinstance(body, dict):
        raise ReviewRejected(400, 'invalid_schema', italian_message('invalid_schema', 'the request must be an object'))
    if body.get('run_id') != run_id:
        raise ReviewRejected(409, 'stale_report',
                             'Il report di questa route è stato rigenerato: ricarica la pagina.')
    confirmed = body.get('confirmed')
    if type(confirmed) is not bool:
        raise ReviewRejected(400, 'invalid_schema', italian_message('invalid_schema', 'confirmed must be true or false'))
    base_revision = body.get('base_revision')
    if base_revision is not None:
        if type(base_revision) is not int or base_revision < 0:
            raise ReviewRejected(400, 'invalid_schema', italian_message('invalid_schema', 'base_revision'))
    frame_key = report_row['frame_key']
    if not confirmed:
        reviewer = body.get('reviewer')
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ReviewRejected(400, 'missing_provenance', italian_message('missing_provenance', 'reviewer'))
        return store.withdraw_review(route_id=route_id, run_id=run_id, frame_key=frame_key,
                                     reviewer=reviewer.strip(), client=client, base_revision=base_revision)
    if not isinstance(body.get('detections'), list):
        raise ReviewRejected(400, 'invalid_schema', italian_message('invalid_schema', 'detections must be a list'))
    frame = build_reviewed_frame(report_row, body, utc_now())
    validate_reviewed_frame(frame, root)
    return store.save_review(route_id=route_id, run_id=run_id, frame_key=frame_key,
                             segment_index=frame['segment_index'], output_index=frame['output_index'],
                             payload=frame, reviewer=frame['reviewer'], client=client,
                             base_revision=base_revision)


def _comparable(row: dict) -> dict:
    """A canonical row without who/when/which file, to compare labels only."""
    result = dict(row)
    result.pop('reviewer', None)
    result.pop('review_timestamp', None)
    provenance = result.get('provenance')
    if isinstance(provenance, dict):
        provenance = dict(provenance)
        provenance.pop('source_review_sha256', None)
        result['provenance'] = provenance
    return result


def _import_state(rows: list[dict], existing: dict) -> str:
    """'new', 'imported' (same labels already in the dataset) or 'changed'."""
    present = 0
    differs = False
    for row in rows:
        old = existing.get(row['annotation_id'])
        if old is None:
            continue
        present = present + 1
        if _comparable(old) != _comparable(row):
            differs = True
    if present == 0:
        return 'new'
    if present == len(rows) and not differs:
        return 'imported'
    return 'changed'


def import_from_db(root: Path, output: Path, *, route: str | None = None, dataset_kind: str = 'gold',
                   store: ReviewStore | None = None) -> dict:
    """Import the confirmed frames of each route's current report from the review database.

    The frames become the same JSONL lines the page export writes and go through
    import_review_sources, so validation, archive and output files are those of
    a JSONL import. Frames whose labels are already in the dataset are skipped,
    so running it again imports only what is new. A frame changed after its
    import is not applied (the canonical dataset is append-only) and is listed.
    """
    from nnslr_tools.preannotate import PreannotationError
    from nnslr_tools.review import latest_report

    if store is None:
        store = ReviewStore(db_path(root))
    if route is not None:
        parse_route_id(route)
        routes = [route]
    else:
        routes = store.route_ids()
    warnings = []
    selected = []
    for route_id in routes:
        try:
            run_id = latest_report(root, route_id).name
        except (PreannotationError, ValueError, OSError) as exc:
            warnings.append(f'{route_id}: no current pre-annotation report ({exc}); its reviews were not imported')
            continue
        for review in store.reviews(route_id, run_id, confirmed_only=True):
            selected.append(review)
        older = store.confirmed_count_in_other_runs(route_id, run_id)
        if older:
            warnings.append(f'{route_id}: {older} confirmed frames belong to older pre-annotation runs '
                            f'and were not imported')
    require(bool(selected), 'empty_dataset', 'no confirmed reviews in the review database')
    existing = {}
    if output.exists():
        for row in load_jsonl(output):
            existing[row.get('annotation_id')] = row
    new_frames = []
    already_imported = 0
    changed = []
    for review in selected:
        rows = reviewed_frame_annotations(review['payload'], PLACEHOLDER_SOURCE_HASH,
                                          dataset_kind=dataset_kind, name='review database')
        state = _import_state(rows, existing)
        if state == 'new':
            new_frames.append(review['payload'])
        elif state == 'imported':
            already_imported = already_imported + 1
        else:
            changed.append({'route_id': review['route_id'], 'run_id': review['run_id'],
                            'frame_key': review['frame_key'], 'reviewer': review['reviewer'],
                            'updated_at': review['updated_at']})
    if changed:
        warnings.append(f'{len(changed)} confirmed frames changed after their import were not applied: '
                        f'the canonical dataset is append-only (see changed_after_import)')
    if not new_frames:
        return {'valid': True, 'source': 'review-db', 'imported_count': 0, 'imported_frame_count': 0,
                'already_imported_frame_count': already_imported, 'changed_after_import': changed,
                'errors': [], 'warnings': warnings, 'output': str(output)}
    report = import_review_sources([('review-db', jsonl_bytes(new_frames))], root, output,
                                   dataset_kind=dataset_kind)
    report['source'] = 'review-db'
    report['imported_frame_count'] = len(new_frames)
    report['already_imported_frame_count'] = already_imported
    report['changed_after_import'] = changed
    report['warnings'] = list(report.get('warnings', [])) + warnings
    return report


__all__ = [
    'DB_RELATIVE_PATH', 'DatasetError', 'ReviewRejected', 'ReviewStore', 'ReviewStoreError', 'SCHEMA_VERSION',
    'build_reviewed_frame', 'db_path', 'import_from_db', 'italian_message', 'public_review',
    'save_frame_review', 'utc_now', 'validate_reviewed_frame',
]
