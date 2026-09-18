from psycopg.conninfo import conninfo_to_dict


def validate_target(dsn):
    name = conninfo_to_dict(dsn).get("dbname", "")
    if name != "qc-researcher" and not name.startswith("qc-researcher-test-"):
        raise ValueError("Use the dedicated qc-researcher database (or qc-researcher-test-* for tests)")
