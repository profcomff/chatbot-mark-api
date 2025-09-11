"""init

Revision ID: 54f6fdbddfce
Revises:
Create Date: 2025-09-12 01:58:50.690957

"""

import sqlalchemy as sa
from alembic import op


revision = '54f6fdbddfce'
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
        sa.Column(
            'is_response_with_buttons',
            sa.Boolean(),
            server_default='false',
            nullable=False,
            comment='Генерировался ли в режиме возврата эндпоинтов (False - значит - чисто генерированный ai ответ)',
        ),
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
