"""initial schema

Revision ID: 0001
Revises: 
Create Date: 2026-09-26 17:40:14.298511+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = '0001'
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('projects',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('name', sa.String(length=200), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_projects')),
    sa.UniqueConstraint('name', name=op.f('uq_projects_name'))
    )
    op.create_table('requirements',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('project_id', sa.Uuid(), nullable=False),
    sa.Column('text', sa.Text(), nullable=False),
    sa.Column('metadata', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    sa.ForeignKeyConstraint(['project_id'], ['projects.id'], name=op.f('fk_requirements_project_id_projects'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_requirements'))
    )
    op.create_index(op.f('ix_requirements_project_id'), 'requirements', ['project_id'], unique=False)
    op.create_table('agent_runs',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('project_id', sa.Uuid(), nullable=False),
    sa.Column('requirement_id', sa.Uuid(), nullable=False),
    sa.Column('status', sa.String(length=32), nullable=False),
    sa.Column('outcome', sa.String(length=32), nullable=True, comment='Final report status once completed'),
    sa.Column('confidence', sa.Float(), nullable=True),
    sa.Column('summary', sa.Text(), nullable=True),
    sa.Column('error_count', sa.Integer(), nullable=False),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('nodes_visited', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('requirement_analysis', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
    sa.Column('root_cause_analysis', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
    sa.Column('final_report', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
    sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.CheckConstraint("status IN ('running', 'awaiting_review', 'completed', 'failed')", name=op.f('ck_agent_runs_status')),
    sa.ForeignKeyConstraint(['project_id'], ['projects.id'], name=op.f('fk_agent_runs_project_id_projects'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['requirement_id'], ['requirements.id'], name=op.f('fk_agent_runs_requirement_id_requirements'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_agent_runs'))
    )
    op.create_index(op.f('ix_agent_runs_project_id'), 'agent_runs', ['project_id'], unique=False)
    op.create_index(op.f('ix_agent_runs_requirement_id'), 'agent_runs', ['requirement_id'], unique=False)
    op.create_index(op.f('ix_agent_runs_status'), 'agent_runs', ['status'], unique=False)
    op.create_table('bug_reports',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('agent_run_id', sa.String(length=64), nullable=False),
    sa.Column('report_key', sa.String(length=32), nullable=False, comment='e.g. BR-1'),
    sa.Column('revision', sa.Integer(), nullable=False),
    sa.Column('title', sa.Text(), nullable=False),
    sa.Column('summary', sa.Text(), nullable=False),
    sa.Column('report_type', sa.String(length=32), nullable=False),
    sa.Column('severity', sa.String(length=16), nullable=False),
    sa.Column('priority', sa.String(length=8), nullable=False),
    sa.Column('affected_component', sa.Text(), nullable=False),
    sa.Column('confidence', sa.Float(), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('related_test_case_ids', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('payload', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False, comment='Full BugReport'),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    sa.CheckConstraint("priority IN ('p0', 'p1', 'p2', 'p3')", name=op.f('ck_bug_reports_priority')),
    sa.CheckConstraint("severity IN ('low', 'medium', 'high', 'critical')", name=op.f('ck_bug_reports_severity')),
    sa.CheckConstraint("status IN ('draft', 'approved', 'rejected', 'superseded')", name=op.f('ck_bug_reports_status')),
    sa.ForeignKeyConstraint(['agent_run_id'], ['agent_runs.id'], name=op.f('fk_bug_reports_agent_run_id_agent_runs'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_bug_reports')),
    sa.UniqueConstraint('agent_run_id', 'report_key', 'revision', name=op.f('uq_bug_reports_agent_run_id_report_key_revision'))
    )
    op.create_index(op.f('ix_bug_reports_agent_run_id'), 'bug_reports', ['agent_run_id'], unique=False)
    op.create_table('human_reviews',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('agent_run_id', sa.String(length=64), nullable=False),
    sa.Column('review_key', sa.String(length=128), nullable=False, comment='HumanReview.review_id'),
    sa.Column('reason', sa.Text(), nullable=False),
    sa.Column('ai_recommendation', sa.Text(), nullable=False),
    sa.Column('evidence', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('confidence', sa.Float(), nullable=False),
    sa.Column('proposed_action', sa.Text(), nullable=False),
    sa.Column('actions', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('sensitive_actions', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('human_decision', sa.String(length=32), nullable=True),
    sa.Column('reviewer', sa.String(length=200), nullable=True),
    sa.Column('reviewer_comment', sa.Text(), nullable=True),
    sa.Column('requested_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('decided_at', sa.DateTime(timezone=True), nullable=True),
    sa.CheckConstraint("human_decision IS NULL OR human_decision IN ('APPROVE', 'REJECT', 'REQUEST_REANALYSIS')", name=op.f('ck_human_reviews_human_decision')),
    sa.ForeignKeyConstraint(['agent_run_id'], ['agent_runs.id'], name=op.f('fk_human_reviews_agent_run_id_agent_runs'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_human_reviews')),
    sa.UniqueConstraint('review_key', name=op.f('uq_human_reviews_review_key'))
    )
    op.create_index(op.f('ix_human_reviews_agent_run_id'), 'human_reviews', ['agent_run_id'], unique=False)
    op.create_table('test_cases',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('agent_run_id', sa.String(length=64), nullable=False),
    sa.Column('requirement_id', sa.Uuid(), nullable=False),
    sa.Column('case_key', sa.String(length=32), nullable=False, comment='e.g. TC-001'),
    sa.Column('title', sa.Text(), nullable=False),
    sa.Column('description', sa.Text(), nullable=False),
    sa.Column('preconditions', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('test_data', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('steps', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('expected_result', sa.Text(), nullable=False),
    sa.Column('priority', sa.String(length=16), nullable=False),
    sa.Column('test_type', sa.String(length=16), nullable=False),
    sa.Column('automation_type', sa.String(length=16), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    sa.ForeignKeyConstraint(['agent_run_id'], ['agent_runs.id'], name=op.f('fk_test_cases_agent_run_id_agent_runs'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['requirement_id'], ['requirements.id'], name=op.f('fk_test_cases_requirement_id_requirements'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_test_cases')),
    sa.UniqueConstraint('agent_run_id', 'case_key', name=op.f('uq_test_cases_agent_run_id_case_key'))
    )
    op.create_index(op.f('ix_test_cases_agent_run_id'), 'test_cases', ['agent_run_id'], unique=False)
    op.create_index(op.f('ix_test_cases_requirement_id'), 'test_cases', ['requirement_id'], unique=False)
    op.create_table('test_runs',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('agent_run_id', sa.String(length=64), nullable=False),
    sa.Column('total', sa.Integer(), nullable=False),
    sa.Column('passed', sa.Integer(), nullable=False),
    sa.Column('failed', sa.Integer(), nullable=False),
    sa.Column('errored', sa.Integer(), nullable=False),
    sa.Column('skipped', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    sa.ForeignKeyConstraint(['agent_run_id'], ['agent_runs.id'], name=op.f('fk_test_runs_agent_run_id_agent_runs'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_test_runs')),
    sa.UniqueConstraint('agent_run_id', name=op.f('uq_test_runs_agent_run_id'))
    )
    op.create_table('test_results',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('test_run_id', sa.Uuid(), nullable=False),
    sa.Column('test_case_id', sa.Uuid(), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('duration_ms', sa.Integer(), nullable=False),
    sa.Column('message', sa.Text(), nullable=True),
    sa.Column('evidence', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    sa.CheckConstraint("status IN ('passed', 'failed', 'error', 'skipped')", name=op.f('ck_test_results_status')),
    sa.ForeignKeyConstraint(['test_case_id'], ['test_cases.id'], name=op.f('fk_test_results_test_case_id_test_cases'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['test_run_id'], ['test_runs.id'], name=op.f('fk_test_results_test_run_id_test_runs'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_test_results')),
    sa.UniqueConstraint('test_run_id', 'test_case_id', name=op.f('uq_test_results_test_run_id_test_case_id'))
    )
    op.create_index(op.f('ix_test_results_test_case_id'), 'test_results', ['test_case_id'], unique=False)
    op.create_index(op.f('ix_test_results_test_run_id'), 'test_results', ['test_run_id'], unique=False)
    op.create_table('failures',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('test_result_id', sa.Uuid(), nullable=False),
    sa.Column('category', sa.String(length=32), nullable=True),
    sa.Column('message', sa.Text(), nullable=True),
    sa.Column('signal', sa.Text(), nullable=True),
    sa.Column('expected_result', sa.Text(), nullable=True),
    sa.Column('actual_result', sa.Text(), nullable=True),
    sa.Column('observed_facts', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    sa.ForeignKeyConstraint(['test_result_id'], ['test_results.id'], name=op.f('fk_failures_test_result_id_test_results'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_failures')),
    sa.UniqueConstraint('test_result_id', name=op.f('uq_failures_test_result_id'))
    )


def downgrade() -> None:
    op.drop_table('failures')
    op.drop_index(op.f('ix_test_results_test_run_id'), table_name='test_results')
    op.drop_index(op.f('ix_test_results_test_case_id'), table_name='test_results')
    op.drop_table('test_results')
    op.drop_table('test_runs')
    op.drop_index(op.f('ix_test_cases_requirement_id'), table_name='test_cases')
    op.drop_index(op.f('ix_test_cases_agent_run_id'), table_name='test_cases')
    op.drop_table('test_cases')
    op.drop_index(op.f('ix_human_reviews_agent_run_id'), table_name='human_reviews')
    op.drop_table('human_reviews')
    op.drop_index(op.f('ix_bug_reports_agent_run_id'), table_name='bug_reports')
    op.drop_table('bug_reports')
    op.drop_index(op.f('ix_agent_runs_status'), table_name='agent_runs')
    op.drop_index(op.f('ix_agent_runs_requirement_id'), table_name='agent_runs')
    op.drop_index(op.f('ix_agent_runs_project_id'), table_name='agent_runs')
    op.drop_table('agent_runs')
    op.drop_index(op.f('ix_requirements_project_id'), table_name='requirements')
    op.drop_table('requirements')
    op.drop_table('projects')
