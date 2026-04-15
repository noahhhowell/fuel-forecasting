import argparse

from cli import load_command
from database import FuelDatabase


class TestLoadCommand:
    """Tests for CLI load command behavior."""

    def test_load_command_replace_prints_backup_summary(self, tmp_path, capsys):
        csv_path = tmp_path / "replacement.csv"
        csv_path.write_text(
            "site_id,grade,day,volume,is_estimated\n"
            "999,UNL,2024-02-01,500.0,0\n"
        )
        db_path = tmp_path / "cli_replace.db"

        seeded = FuelDatabase(str(db_path))
        try:
            seeded.conn.execute(
                "INSERT INTO sales (site_id, grade, day, volume, is_estimated) "
                "VALUES ('111', 'UNL', '2024-01-01', 100.0, 0)"
            )
            seeded.conn.commit()
        finally:
            seeded.close()

        args = argparse.Namespace(
            command="load",
            database=str(db_path),
            header_row=4,
            file=str(csv_path),
            directory=None,
            replace=True,
        )

        load_command(args)
        output = capsys.readouterr().out

        assert "Backup created:" in output
        assert "Calibration cleared: Yes" in output

    def test_load_command_rejects_replace_with_directory(self, tmp_path):
        args = argparse.Namespace(
            command="load",
            database=str(tmp_path / "db.db"),
            header_row=4,
            file=None,
            directory=str(tmp_path),
            replace=True,
        )

        try:
            load_command(args)
        except ValueError as exc:
            assert "--replace can only be used with --file" in str(exc)
        else:
            raise AssertionError("Expected load_command to reject --replace with --directory")
