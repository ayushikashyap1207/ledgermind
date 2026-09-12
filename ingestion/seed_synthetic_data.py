"""
Populates the schema with synthetic-but-internally-consistent financial
data: real double-entry bookkeeping (every transaction has matching
debit/credit legs that sum to zero), plausible vendors/invoices, and
enough volume (5k+ rows total) for aggregate SQL questions ("total
Q3 revenue", "top vendor by spend") to be meaningful rather than
trivial.

Run: python -m ingestion.seed_synthetic_data
"""

from __future__ import annotations

import datetime as dt
import random

from app.sql.db import get_engine
from app.sql.models import Account, Base, GeneralLedger, Invoice, Transaction, Vendor
from sqlalchemy.orm import Session

random.seed(42)

ACCOUNTS = [
    ("1000", "Cash and Equivalents", "asset"),
    ("1100", "Accounts Receivable", "asset"),
    ("1200", "Inventory", "asset"),
    ("2000", "Accounts Payable", "liability"),
    ("2100", "Accrued Liabilities", "liability"),
    ("3000", "Common Stock", "equity"),
    ("3100", "Retained Earnings", "equity"),
    ("4000", "Product Revenue", "revenue"),
    ("4100", "Services Revenue", "revenue"),
    ("5000", "Cost of Goods Sold", "expense"),
    ("5100", "R&D Expense", "expense"),
    ("5200", "Sales & Marketing Expense", "expense"),
    ("5300", "General & Administrative Expense", "expense"),
]

VENDOR_CATEGORIES = ["Cloud Infrastructure", "Office Supplies", "Professional Services",
                      "Marketing", "Travel", "Software Licensing", "Facilities"]
COUNTRIES = ["United States", "India", "United Kingdom", "Germany", "Singapore", "Canada"]

VENDOR_NAMES = [
    "Northwind Cloud Services", "BluePeak Consulting", "Horizon Office Supply",
    "Vertex Marketing Group", "Skyline Travel Partners", "Cedarline Software",
    "Meridian Facilities Mgmt", "Atlas Analytics Co", "Fernbridge Legal Services",
    "Quantum Hosting Inc", "Ridgeline Hardware", "Solstice Design Studio",
    "Ironwood Logistics", "Palisade Security Systems", "Wavelength Media",
]

TXN_DESCRIPTIONS = [
    "Customer payment received", "Vendor invoice payment", "Payroll disbursement",
    "Cloud infra subscription", "Product sale - enterprise tier", "Product sale - SMB tier",
    "Consulting services revenue", "Office rent payment", "Marketing campaign spend",
    "Software license renewal", "Travel & expense reimbursement", "Equipment purchase",
]


def _fiscal_quarter(d: dt.date) -> str:
    q = (d.month - 1) // 3 + 1
    return f"{d.year}-Q{q}"


def _random_date(start: dt.date, end: dt.date) -> dt.date:
    delta = (end - start).days
    return start + dt.timedelta(days=random.randint(0, delta))


def seed(n_transactions: int = 1800, n_invoices: int = 600) -> dict:
    engine = get_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    start = dt.date(2024, 1, 1)
    end = dt.date(2025, 12, 31)

    with Session(engine) as session:
        accounts = [Account(account_code=c, account_name=n, account_type=t) for c, n, t in ACCOUNTS]
        session.add_all(accounts)
        session.flush()

        by_type = {}
        for a in accounts:
            by_type.setdefault(a.account_type, []).append(a)

        vendors = [
            Vendor(
                vendor_name=name,
                vendor_category=random.choice(VENDOR_CATEGORIES),
                country=random.choice(COUNTRIES),
            )
            for name in VENDOR_NAMES
        ]
        session.add_all(vendors)
        session.flush()

        # --- Transactions + balanced double-entry ledger legs ---
        ledger_rows = 0
        for i in range(n_transactions):
            t_date = _random_date(start, end)
            amount = round(random.uniform(150, 85000), 2)
            desc = random.choice(TXN_DESCRIPTIONS)

            if "revenue" in desc.lower() or "sale" in desc.lower() or "payment received" in desc.lower():
                debit_acct = random.choice(by_type["asset"])
                credit_acct = random.choice(by_type["revenue"])
            else:
                debit_acct = random.choice(by_type["expense"] + by_type["liability"])
                credit_acct = random.choice(by_type["asset"])

            txn = Transaction(
                transaction_date=t_date,
                description=desc,
                amount=amount,
                account_id=debit_acct.account_id,
                status=random.choices(["posted", "pending", "void"], weights=[92, 6, 2])[0],
            )
            session.add(txn)
            session.flush()

            fq = _fiscal_quarter(t_date)
            session.add_all([
                GeneralLedger(transaction_id=txn.transaction_id, account_id=debit_acct.account_id,
                               entry_date=t_date, debit=amount, credit=0, fiscal_quarter=fq),
                GeneralLedger(transaction_id=txn.transaction_id, account_id=credit_acct.account_id,
                               entry_date=t_date, debit=0, credit=amount, fiscal_quarter=fq),
            ])
            ledger_rows += 2

        # --- Invoices ---
        for i in range(n_invoices):
            issue = _random_date(start, end)
            due = issue + dt.timedelta(days=random.choice([15, 30, 45, 60]))
            vendor = random.choice(vendors)
            status = random.choices(["paid", "overdue", "pending"], weights=[70, 12, 18])[0]
            session.add(Invoice(
                invoice_number=f"INV-{2024000 + i}",
                vendor_id=vendor.vendor_id,
                issue_date=issue,
                due_date=due,
                amount=round(random.uniform(200, 42000), 2),
                status=status,
                line_item_description=f"{vendor.vendor_category} services — {vendor.vendor_name}",
            ))

        session.commit()

        counts = {
            "accounts": len(accounts),
            "vendors": len(vendors),
            "transactions": n_transactions,
            "invoices": n_invoices,
            "general_ledger": ledger_rows,
        }
        counts["total"] = sum(counts.values())
        return counts


if __name__ == "__main__":
    counts = seed()
    print("Seeded synthetic data:")
    for k, v in counts.items():
        print(f"  {k}: {v}")
