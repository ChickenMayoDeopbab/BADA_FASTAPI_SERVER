from sqlalchemy.orm.attributes import Mapped


def create_feedback(
        silence_total: float,
        scenario_id: Mapped[int] = None
) -> bool:
