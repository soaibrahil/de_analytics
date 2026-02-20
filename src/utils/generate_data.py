"""
Healthcare Data Generation Module
Generates synthetic data for medallion architecture layers.
"""

import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import typer
from dataclasses import dataclass

app = typer.Typer()


@dataclass
class DataGeneratorConfig:
    """Configuration for data generation."""
    num_patients: int = 50000
    num_providers: int = 2000
    num_facilities: int = 100
    num_payers: int = 20
    encounters_per_patient: Tuple[int, int] = (2, 20)
    max_diag_per_enc: int = 5
    max_proc_per_enc: int = 6
    start_date: datetime = datetime(2019, 1, 1)
    end_date: datetime = datetime(2024, 12, 31)
    random_seed: int = 42
    output_dir: Path = Path(".")


class DateDimensionGenerator:
    """Generates date dimension table."""

    def __init__(self, start_date: datetime, end_date: datetime):
        self.start_date = start_date
        self.end_date = end_date

    def generate(self) -> pd.DataFrame:
        """Generate date dimension."""
        dates = pd.date_range(self.start_date, self.end_date)
        df = pd.DataFrame({
            "date_key": dates.strftime("%Y%m%d").astype(int),
            "date": dates,
            "year": dates.year,
            "quarter": dates.quarter,
            "month": dates.month,
            "day": dates.day,
            "week": dates.isocalendar().week
        })
        return df


class PatientDimensionGenerator:
    """Generates patient dimension table."""

    def __init__(self, num_patients: int, start_date: datetime):
        self.num_patients = num_patients
        self.start_date = start_date
        self.fake = Faker()

    def generate(self) -> pd.DataFrame:
        """Generate patient dimension."""
        data = []
        for i in range(self.num_patients):
            data.append({
                "patient_key": i + 1,
                "patient_id": f"P{i+1:07}",
                "first_name": self.fake.first_name(),
                "last_name": self.fake.last_name(),
                "gender": random.choice(["M", "F"]),
                "birth_date": self.fake.date_of_birth(minimum_age=0, maximum_age=90),
                "city": self.fake.city(),
                "state": self.fake.state(),
                "zip": self.fake.zipcode(),
                "effective_from": self.start_date,
                "effective_to": None,
                "is_current": 1
            })
        return pd.DataFrame(data)


class ProviderDimensionGenerator:
    """Generates provider dimension table."""

    SPECIALTIES = [
        "Cardiology", "Orthopedics", "Oncology", "Radiology",
        "Emergency", "Pediatrics", "Neurology"
    ]
    DEPARTMENTS = ["Inpatient", "Outpatient", "ER"]

    def __init__(self, num_providers: int, start_date: datetime):
        self.num_providers = num_providers
        self.start_date = start_date
        self.fake = Faker()

    def generate(self) -> pd.DataFrame:
        """Generate provider dimension."""
        data = []
        for i in range(self.num_providers):
            data.append({
                "provider_key": i + 1,
                "provider_id": f"PR{i+1:05}",
                "provider_name": self.fake.name(),
                "specialty": random.choice(self.SPECIALTIES),
                "department": random.choice(self.DEPARTMENTS),
                "effective_from": self.start_date,
                "effective_to": None,
                "is_current": 1
            })
        return pd.DataFrame(data)


class PayerDimensionGenerator:
    """Generates payer dimension table."""

    PAYER_TYPES = ["Commercial", "Medicare", "Medicaid", "Self-Pay"]

    def __init__(self, num_payers: int):
        self.num_payers = num_payers

    def generate(self) -> pd.DataFrame:
        """Generate payer dimension."""
        data = []
        for i in range(self.num_payers):
            data.append({
                "payer_key": i + 1,
                "payer_name": f"Payer_{i+1}",
                "payer_type": random.choice(self.PAYER_TYPES)
            })
        return pd.DataFrame(data)


class FacilityDimensionGenerator:
    """Generates facility dimension table."""

    def __init__(self, num_facilities: int):
        self.num_facilities = num_facilities
        self.fake = Faker()

    def generate(self) -> pd.DataFrame:
        """Generate facility dimension."""
        data = []
        for i in range(self.num_facilities):
            data.append({
                "facility_key": i + 1,
                "facility_name": f"Hospital_{i+1}",
                "city": self.fake.city(),
                "state": self.fake.state(),
                "bed_count": random.randint(50, 800)
            })
        return pd.DataFrame(data)


class CodeGenerator:
    """Generates medical codes."""

    @staticmethod
    def generate_icd_codes(n: int = 200) -> List[str]:
        """Generate ICD diagnosis codes."""
        return [f"I{random.randint(10,99)}.{random.randint(0,9)}" for _ in range(n)]

    @staticmethod
    def generate_cpt_codes(n: int = 300) -> List[str]:
        """Generate CPT procedure codes."""
        return [f"{random.randint(10000,69999)}" for _ in range(n)]


class EncounterFactGenerator:
    """Generates encounter fact table."""

    def __init__(self, config: DataGeneratorConfig, dim_patient: pd.DataFrame,
                 dim_provider: pd.DataFrame, dim_facility: pd.DataFrame,
                 dim_payer: pd.DataFrame):
        self.config = config
        self.dim_patient = dim_patient
        self.dim_provider = dim_provider
        self.dim_facility = dim_facility
        self.dim_payer = dim_payer
        self.fake = Faker()

    def generate(self) -> Tuple[pd.DataFrame, list]:
        """Generate encounter fact table and transaction rows."""
        encounter_rows = []
        transaction_rows = []
        encounter_key = 1
        transaction_key = 1

        for _, patient in self.dim_patient.iterrows():
            num_enc = random.randint(*self.config.encounters_per_patient)
            for _ in range(num_enc):
                encounter_date = self.fake.date_between(
                    self.config.start_date, self.config.end_date
                )
                date_key = int(encounter_date.strftime("%Y%m%d"))
                provider_key = random.randint(1, self.config.num_providers)
                facility_key = random.randint(1, self.config.num_facilities)
                payer_key = random.randint(1, self.config.num_payers)

                total_charge = round(np.random.gamma(5, 500), 2)
                allowed = round(total_charge * random.uniform(0.6, 0.9), 2)
                paid = round(allowed * random.uniform(0.7, 1.0), 2)
                adjustment = round(total_charge - paid, 2)

                encounter_rows.append({
                    "encounter_key": encounter_key,
                    "patient_key": patient["patient_key"],
                    "provider_key": provider_key,
                    "facility_key": facility_key,
                    "payer_key": payer_key,
                    "date_key": date_key,
                    "drg_code": f"DRG{random.randint(100,999)}",
                    "total_charge": total_charge,
                    "total_allowed": allowed,
                    "total_paid": paid,
                    "total_adjustment": adjustment
                })

                # Generate transactions
                transaction_rows.extend([
                    {
                        "transaction_key": transaction_key,
                        "encounter_key": encounter_key,
                        "transaction_type": "CHARGE",
                        "amount": total_charge,
                        "date_key": date_key
                    },
                    {
                        "transaction_key": transaction_key + 1,
                        "encounter_key": encounter_key,
                        "transaction_type": "PAYMENT",
                        "amount": paid,
                        "date_key": date_key
                    },
                    {
                        "transaction_key": transaction_key + 2,
                        "encounter_key": encounter_key,
                        "transaction_type": "ADJUSTMENT",
                        "amount": adjustment,
                        "date_key": date_key
                    }
                ])
                transaction_key += 3
                encounter_key += 1

        return pd.DataFrame(encounter_rows), transaction_rows


class HealthcareDataGenerator:
    """Main class for healthcare data generation."""

    def __init__(self, config: DataGeneratorConfig):
        self.config = config
        self._set_random_seed()

    def _set_random_seed(self):
        """Set random seed for reproducibility."""
        np.random.seed(self.config.random_seed)
        random.seed(self.config.random_seed)

    def generate_all(self) -> dict:
        """Generate all dimensions and fact tables."""
        typer.echo("🔄 Generating dimensions...")

        # Generate dimensions
        dim_date = DateDimensionGenerator(
            self.config.start_date, self.config.end_date
        ).generate()
        dim_patient = PatientDimensionGenerator(
            self.config.num_patients, self.config.start_date
        ).generate()
        dim_provider = ProviderDimensionGenerator(
            self.config.num_providers, self.config.start_date
        ).generate()
        dim_payer = PayerDimensionGenerator(self.config.num_payers).generate()
        dim_facility = FacilityDimensionGenerator(
            self.config.num_facilities
        ).generate()

        typer.echo("🔄 Generating facts...")

        # Generate facts
        fact_encounter, transaction_rows = EncounterFactGenerator(
            self.config, dim_patient, dim_provider, dim_facility, dim_payer
        ).generate()
        fact_transaction = pd.DataFrame(transaction_rows)

        return {
            "dim_date": dim_date,
            "dim_patient": dim_patient,
            "dim_provider": dim_provider,
            "dim_payer": dim_payer,
            "dim_facility": dim_facility,
            "fact_encounter": fact_encounter,
            "fact_transaction": fact_transaction
        }

    def save_to_bronze(self, data: dict):
        """Save data to bronze layer."""
        output_dir = self.config.output_dir / "data" / "bronze"
        output_dir.mkdir(parents=True, exist_ok=True)

        data["fact_encounter"].to_parquet(output_dir / "bronze_fact_encounter.parquet")
        data["fact_transaction"].to_parquet(output_dir / "bronze_fact_transaction.parquet")
        data["dim_patient"].to_parquet(output_dir / "bronze_dim_patient.parquet")
        data["dim_provider"].to_parquet(output_dir / "bronze_dim_provider.parquet")
        data["dim_payer"].to_parquet(output_dir / "bronze_dim_payer.parquet")
        data["dim_facility"].to_parquet(output_dir / "bronze_dim_facility.parquet")
        data["dim_date"].to_parquet(output_dir / "bronze_dim_date.parquet")

        typer.echo(f"✅ Bronze layer saved to {output_dir}")

    def save_to_silver(self, data: dict):
        """Save cleaned data to silver layer."""
        output_dir = self.config.output_dir / "data" / "silver"
        output_dir.mkdir(parents=True, exist_ok=True)

        data["fact_encounter"].to_parquet(output_dir / "silver_fact_encounter.parquet")
        data["fact_transaction"].to_parquet(output_dir / "silver_fact_transaction.parquet")

        typer.echo(f"✅ Silver layer saved to {output_dir}")

    def generate_gold_aggregations(self, data: dict) -> dict:
        """Generate gold layer aggregations."""
        gold_provider_perf = (
            data["fact_encounter"]
            .groupby("provider_key")
            .agg(
                total_revenue=("total_paid", "sum"),
                total_encounters=("encounter_key", "count")
            )
            .reset_index()
        )

        gold_payer_perf = (
            data["fact_encounter"]
            .groupby("payer_key")
            .agg(
                total_revenue=("total_paid", "sum"),
                claim_count=("encounter_key", "count")
            )
            .reset_index()
        )

        return {
            "provider_performance": gold_provider_perf,
            "payer_performance": gold_payer_perf
        }

    def save_to_gold(self, gold_data: dict):
        """Save aggregated data to gold layer."""
        output_dir = self.config.output_dir / "data" / "gold"
        output_dir.mkdir(parents=True, exist_ok=True)

        gold_data["provider_performance"].to_parquet(
            output_dir / "gold_provider_performance.parquet"
        )
        gold_data["payer_performance"].to_parquet(
            output_dir / "gold_payer_performance.parquet"
        )

        typer.echo(f"✅ Gold layer saved to {output_dir}")


@app.command()
def generate(
    patients: int = typer.Option(50000, "--patients", "-p", help="Number of patients"),
    providers: int = typer.Option(2000, "--providers", "-pr", help="Number of providers"),
    facilities: int = typer.Option(100, "--facilities", "-f", help="Number of facilities"),
    payers: int = typer.Option(20, "--payers", "-py", help="Number of payers"),
    output_path: str = typer.Option(".", "--output", "-o", help="Output directory"),
    all_layers: bool = typer.Option(True, "--all-layers", "-a", help="Generate all layers"),
    bronze_only: bool = typer.Option(False, "--bronze", "-b", help="Bronze layer only"),
    silver_only: bool = typer.Option(False, "--silver", "-s", help="Silver layer only"),
):
    """Generate synthetic healthcare data for all medallion layers."""
    config = DataGeneratorConfig(
        num_patients=patients,
        num_providers=providers,
        num_facilities=facilities,
        num_payers=payers,
        output_dir=Path(output_path)
    )

    generator = HealthcareDataGenerator(config)
    data = generator.generate_all()

    if bronze_only or all_layers:
        generator.save_to_bronze(data)

    if silver_only or all_layers:
        generator.save_to_silver(data)

    if not bronze_only and not silver_only and all_layers:
        gold_data = generator.generate_gold_aggregations(data)
        generator.save_to_gold(gold_data)

    typer.echo("✨ Healthcare data generation complete!")
    typer.echo(f"📊 Generated {patients} patients with {len(data['fact_encounter'])} encounters")


@app.command()
def info():
    """Display generator information and default configuration."""
    config = DataGeneratorConfig()
    typer.echo("\n📋 Data Generator Configuration:\n")
    typer.echo(f"  Patients:              {config.num_patients:,}")
    typer.echo(f"  Providers:             {config.num_providers:,}")
    typer.echo(f"  Facilities:            {config.num_facilities:,}")
    typer.echo(f"  Payers:                {config.num_payers:,}")
    typer.echo(f"  Encounters per patient: {config.encounters_per_patient}")
    typer.echo(f"  Date range:            {config.start_date.date()} to {config.end_date.date()}")
    typer.echo()


if __name__ == "__main__":
    app()