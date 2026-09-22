import os
from typing import Literal, Mapping

from sqlalchemy import (
    Column,
    Double,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.types import LargeBinary, TypeDecorator


# ============================================================
# 16-byte hash 主键
# ============================================================

class HashBinary(TypeDecorator):
    """
    只接受 atom_hash/bond_hash 等函数返回的 16-byte bytes。

    不转换字符串。
    不 packbits。
    不再次计算 MD5。
    """
    impl = LargeBinary
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None

        if not isinstance(value, bytes):
            raise TypeError(
                f"hash_str must be bytes, got {type(value).__name__}"
            )

        if len(value) != 16:
            raise ValueError(
                f"hash_str must contain exactly 16 bytes, "
                f"got {len(value)}"
            )

        return value

    def process_result_value(self, value, dialect):
        if value is None:
            return None

        return bytes(value)


Base = declarative_base()


class AtomType(Base):
    __tablename__ = "atom_type"

    hash_str = Column(
        HashBinary(16),
        primary_key=True,
        nullable=False,
    )

    opls_num  = Column(String(10) , nullable = False)
    element   = Column(String(10) , nullable = False)
    mass      = Column(Double)
    charge    = Column(Double)
    sigma     = Column(Double)
    epsilon   = Column(Double)
    ptype     = Column(String(10))
    bond_type = Column(String(10))


class BondType(Base):
    __tablename__ = "bond_type"

    hash_str = Column(
        HashBinary(16),
        primary_key=True,
        nullable=False,
    )

    opls_i = Column(String(10), nullable=False)
    opls_j = Column(String(10), nullable=False)

    k = Column(Double)
    r0 = Column(Double)
    ftype = Column(Integer)


class AngleType(Base):
    __tablename__ = "angle_type"

    hash_str = Column(
        HashBinary(16),
        primary_key=True,
        nullable=False,
    )

    opls_i = Column(String(10), nullable=False)
    opls_j = Column(String(10), nullable=False)
    opls_k = Column(String(10), nullable=False)

    k = Column(Double)
    t0 = Column(Double)
    ftype = Column(Integer)


class DihedralType(Base):
    __tablename__ = "dihedral_type"

    hash_str = Column(
        HashBinary(16),
        primary_key=True,
        nullable=False,
    )

    opls_i = Column(String(10), nullable=False)
    opls_j = Column(String(10), nullable=False)
    opls_k = Column(String(10), nullable=False)
    opls_l = Column(String(10), nullable=False)

    C0 = Column(Double)
    C1 = Column(Double)
    C2 = Column(Double)
    C3 = Column(Double)
    C4 = Column(Double)
    C5 = Column(Double)

    ftype = Column(Integer)


class ImproperType(Base):
    __tablename__ = "improper_type"

    hash_str = Column(
        HashBinary(16),
        primary_key=True,
        nullable=False,
    )

    opls_i = Column(String(10), nullable=False)
    opls_j = Column(String(10), nullable=False)
    opls_k = Column(String(10), nullable=False)
    opls_l = Column(String(10), nullable=False)

    k = Column(Double)
    psi0 = Column(Double)
    ftype = Column(Integer)


class AmberDB:
    def __init__(
            self,
            db_path: str = "amber.db",
            overwrite: bool = False,
    ):
        if overwrite and os.path.isfile(db_path):
            os.remove(db_path)

        uri = f"sqlite:///{os.path.abspath(db_path)}"

        self.engine = create_engine(
            uri,
            echo=False,
            future=True,
        )

        Base.metadata.create_all(self.engine)

        Session = sessionmaker(
            bind=self.engine,
            future=True,
            expire_on_commit=False,
        )

        self.session = Session()

    def _insert(self, obj) -> bool:

        self.session.add(obj)

        try:
            self.session.commit()
            return True
        except IntegrityError:
            self.session.rollback()
            return False

    def insert(self, data: Mapping[bytes, Base]) -> None:

        if not data:
            return

        self.session.add_all(list(data.values()))

        try:
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise

    def search(
            self,
            target: Literal[
                "atom",
                "bond",
                "angle",
                "dihedral",
                "improper",
            ],
            **kw,
    ) -> list:
        models = {
            "atom"    : AtomType,
            "bond"    : BondType,
            "angle"   : AngleType,
            "dihedral": DihedralType,
            "improper": ImproperType,
        }

        try:
            model = models[target]
        except KeyError:
            raise ValueError(f"unknown search target: {target!r}")

        query = self.session.query(model)

        for name, value in kw.items():
            if not hasattr(model, name):
                raise ValueError(
                    f"{model.__name__} has no column {name!r}"
                )

            query = query.filter(getattr(model, name) == value)

        return query.all()

    def stat(self) -> dict[str, int]:
        models = {
            "atom"    : AtomType,
            "bond"    : BondType,
            "angle"   : AngleType,
            "dihedral": DihedralType,
            "improper": ImproperType,
        }

        result = {
            name: self.session.query(model).count()
            for name, model in models.items()
        }

        for name, count in result.items():
            print(f"{name}: {count}")

        return result

    def close(self) -> None:
        self.session.close()
        self.engine.dispose()
