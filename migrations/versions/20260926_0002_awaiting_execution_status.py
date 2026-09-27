"""agent run status awaiting_execution

A run can pause after test planning until someone starts test execution.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-26 23:30:00+00:00
"""

from collections.abc import Sequence

from alembic import op

revision: str = '0002'
down_revision: str | None = '0001'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _replace_status_check(allowed: str) -> None:
    # Batch mode: SQLite cannot ALTER constraints, so the table is recreated there;
    # on PostgreSQL this is a plain DROP/ADD CONSTRAINT.
    with op.batch_alter_table('agent_runs') as batch_op:
        batch_op.drop_constraint(op.f('ck_agent_runs_status'), type_='check')
        batch_op.create_check_constraint(op.f('ck_agent_runs_status'), f"status IN ({allowed})")


def upgrade() -> None:
    _replace_status_check("'running', 'awaiting_execution', 'awaiting_review', 'completed', 'failed'")


def downgrade() -> None:
    op.execute("UPDATE agent_runs SET status = 'running' WHERE status = 'awaiting_execution'")
    _replace_status_check("'running', 'awaiting_review', 'completed', 'failed'")
