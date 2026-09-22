import os
from typing import Literal, Mapping, Optional

from sqlalchemy import (
    Column,
    Double,
    Integer,
    String,
    JSON,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.types import LargeBinary, TypeDecorator


# ============================================================
# 16-byte hash Primary Key Type
# ============================================================
class HashBinary(TypeDecorator):
    """
    Accepts only 16-byte bytes returned by hash functions.
    No string conversion. No packbits. No MD5 recalculation.
    """
    impl = LargeBinary
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if not isinstance(value, bytes):
            raise TypeError(f"hash_str must be bytes, got {type(value).__name__}")
        if len(value) != 16:
            raise ValueError(f"hash_str must contain exactly 16 bytes, got {len(value)}")
        return value

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return bytes(value)


Base = declarative_base()


# ============================================================
# Core Database Models (2-Table Architecture)
# ============================================================
class AtomType(Base):
    """
    Universal table for all Atom properties.
    """
    __tablename__ = "atom_type"

    hash_str = Column(HashBinary(16), primary_key=True, nullable=False)

    element = Column(String(10), nullable=False)
    mass = Column(Double, nullable=False)
    charge = Column(Double, nullable=False)
    type_label = Column(String(30), nullable=False)

    # Dynamic parameter payload (e.g., LJ, Morse)
    params = Column(JSON, nullable=False)

    def __repr__(self):
        """Make print(atom) human-readable."""
        h = self.hash_str.hex()[:8] if self.hash_str else "None"
        return (f"<AtomType | Hash:{h}... | Elem:{self.element} | "
                f"Label:'{self.type_label}' | Params:{self.params}>")


class BondedType(Base):
    """
    Universal table for Bond, Angle, Dihedral, and Improper.
    """
    __tablename__ = "bonded_type"

    hash_str = Column(HashBinary(16), primary_key=True, nullable=False)

    # e.g., "bond", "angle", "dihedral", "improper"
    interaction_type = Column(String(20), nullable=False, index=True)

    # e.g., ["CT", "CT"] or ["CT", "CT", "CT", "CT"]
    type_labels = Column(JSON, nullable=False)

    ftype = Column(Integer, default=1)

    # Dynamic parameter payload
    params = Column(JSON, nullable=False)

    def __repr__(self):
        """Make print(bonded) human-readable."""
        h = self.hash_str.hex()[:8] if self.hash_str else "None"
        # Join list into a string like "CT-CT-CT" for better display
        labels = "-".join(self.type_labels) if self.type_labels else ""
        return (f"<{self.interaction_type.capitalize()}Type | Hash:{h}... | "
                f"Types:[{labels}] | ftype:{self.ftype} | Params:{self.params}>")


# ============================================================
# Database Manager
# ============================================================
class ForceFieldDB:
    def __init__(
            self,
            db_path: str,
            overwrite: bool = False,
    ):
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        if overwrite and os.path.isfile(db_path):
            os.remove(db_path)

        uri = f"sqlite:///{os.path.abspath(db_path)}"
        self.engine = create_engine(uri, echo=False, future=True)
        Base.metadata.create_all(self.engine)

        Session = sessionmaker(bind=self.engine, future=True, expire_on_commit=False)
        self.session = Session()

    def insert(self, data: Mapping[bytes, Base]) -> None:
        if not data:
            return
        self.session.add_all(list(data.values()))
        try:
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise

    def get_by_hash(
            self,
            target: Literal["atom", "bonded"],
            hash_val: bytes
    ) -> Optional[Base]:
        """O(1) Hash Query."""
        model = AtomType if target == "atom" else BondedType
        return self.session.query(model).filter(model.hash_str == hash_val).first()

    def search(
            self,
            target: Literal["atom", "bond", "angle", "dihedral", "improper"],
            **kw,
    ) -> list:
        if target == "atom":
            model = AtomType
            query = self.session.query(model)
        else:
            model = BondedType
            query = self.session.query(model).filter(model.interaction_type == target)

        for name, value in kw.items():
            if not hasattr(model, name):
                raise ValueError(f"{model.__name__} has no column {name!r}")
            query = query.filter(getattr(model, name) == value)

        return query.all()

    def stat(self) -> dict[str, int]:
        result = {
            "atom": self.session.query(AtomType).count(),
            "bond": self.session.query(BondedType).filter_by(interaction_type="bond").count(),
            "angle": self.session.query(BondedType).filter_by(interaction_type="angle").count(),
            "dihedral": self.session.query(BondedType).filter_by(interaction_type="dihedral").count(),
            "improper": self.session.query(BondedType).filter_by(interaction_type="improper").count(),
        }
        for name, count in result.items():
            print(f"{name}: {count}")
        return result

    def close(self) -> None:
        self.session.close()
        self.engine.dispose()
