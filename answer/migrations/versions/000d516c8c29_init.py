"""init

Revision ID: 000d516c8c29
Revises:
Create Date: 2025-09-08 21:33:09.976799

"""

import sqlalchemy as sa
from alembic import op


revision = '000d516c8c29'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'user',
        sa.Column('id', sa.Integer(), nullable=False, comment='Идентификатор пользователя'),
        sa.Column('chat_id', sa.String(), nullable=False, comment='Тг айди чата с пользователем'),
        sa.Column('create_ts', sa.DateTime(), nullable=False, comment='Таймстемп создания пользователя'),
        sa.Column('is_deleted', sa.Boolean(), server_default='false', nullable=False, comment='Флаг софтделита'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('chat_id'),
    )
    op.create_table(
        'conversation',
        sa.Column('id', sa.Integer(), nullable=False, comment='Идентификатор записи диалога'),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('request', sa.String(), server_default='request_text', nullable=False, comment='Строка запроса'),
        sa.Column('response', sa.String(), server_default='response_text', nullable=False, comment='Строка ответа'),
        sa.Column('create_ts', sa.DateTime(), nullable=False, comment='Таймстемп создания пары request/response'),
        sa.Column('is_deleted', sa.Boolean(), server_default='false', nullable=False, comment='Флаг софтделита'),
        sa.ForeignKeyConstraint(
            ['user_id'],
            ['user.id'],
        ),
        sa.PrimaryKeyConstraint('id'),
    )


def downgrade():
    op.drop_table('conversation')
    op.drop_table('user')
