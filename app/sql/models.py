"""
Realistic-enough financial schema: accounts, transactions, invoices,
general_ledger, vendors. This is also the source of truth for the
table/column allowlist used by the safety validator — see
app/sql/schema_registry.py, which introspects these models rather than
hand-maintaining a parallel list that could drift.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import Date, ForeignKey, Numeric, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Account(Base):
    __tablename__ = "accounts"

    account_id: Mapped[int] = mapped_column(primary_key=True)
    account_code: Mapped[str] = mapped_column(String(20), unique=True)
    account_name: Mapped[str] = mapped_column(String(100))
    account_type: Mapped[str] = mapped_column(String(30))  # asset/liability/equity/revenue/expense
    currency: Mapped[str] = mapped_column(String(3), default="USD")

    ledger_entries: Mapped[list["GeneralLedger"]] = relationship(back_populates="account")


class Vendor(Base):
    __tablename__ = "vendors"

    vendor_id: Mapped[int] = mapped_column(primary_key=True)
    vendor_name: Mapped[str] = mapped_column(String(150))
    vendor_category: Mapped[str] = mapped_column(String(50))
    country: Mapped[str] = mapped_column(String(56))

    invoices: Mapped[list["Invoice"]] = relationship(back_populates="vendor")


class Transaction(Base):
    __tablename__ = "transactions"

    transaction_id: Mapped[int] = mapped_column(primary_key=True)
    transaction_date: Mapped[dt.date] = mapped_column(Date)
    description: Mapped[str] = mapped_column(String(255))
    amount: Mapped[float] = mapped_column(Numeric(14, 2))
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.account_id"))
    status: Mapped[str] = mapped_column(String(20), default="posted")  # posted/pending/void

    ledger_entries: Mapped[list["GeneralLedger"]] = relationship(back_populates="transaction")


class Invoice(Base):
    __tablename__ = "invoices"

    invoice_id: Mapped[int] = mapped_column(primary_key=True)
    invoice_number: Mapped[str] = mapped_column(String(30), unique=True)
    vendor_id: Mapped[int] = mapped_column(ForeignKey("vendors.vendor_id"))
    issue_date: Mapped[dt.date] = mapped_column(Date)
    due_date: Mapped[dt.date] = mapped_column(Date)
    amount: Mapped[float] = mapped_column(Numeric(14, 2))
    status: Mapped[str] = mapped_column(String(20))  # paid/overdue/pending
    line_item_description: Mapped[str] = mapped_column(Text)

    vendor: Mapped["Vendor"] = relationship(back_populates="invoices")


class GeneralLedger(Base):
    """Double-entry journal: one row per debit or credit leg."""

    __tablename__ = "general_ledger"

    entry_id: Mapped[int] = mapped_column(primary_key=True)
    transaction_id: Mapped[int] = mapped_column(ForeignKey("transactions.transaction_id"))
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.account_id"))
    entry_date: Mapped[dt.date] = mapped_column(Date)
    debit: Mapped[float] = mapped_column(Numeric(14, 2), default=0)
    credit: Mapped[float] = mapped_column(Numeric(14, 2), default=0)
    fiscal_quarter: Mapped[str] = mapped_column(String(7))  # e.g. "2025-Q3"

    account: Mapped["Account"] = relationship(back_populates="ledger_entries")
    transaction: Mapped["Transaction"] = relationship(back_populates="ledger_entries")
