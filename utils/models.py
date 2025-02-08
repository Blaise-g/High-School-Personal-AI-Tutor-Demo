# utils/models.py

from pydantic import BaseModel, Field
from typing import List, Optional

class DiaryEntry(BaseModel):
    tipo: str = Field(
        ...,
        description="Tipo di voce, dovrebbe essere 'Interrogazione', 'Verifica', 'Compito', 'Argomento'"
    )
    data: str = Field(
        ...,
        description="Data nel formato YYYY-MM-DD"
    )
    materia: str = Field(
        ...,
        description="Materia"
    )
    argomenti: List[str] = Field(
        ...,
        description="Lista degli argomenti trattati"
    )
    data_scadenza: Optional[str] = Field(
        None,
        description="Data di scadenza per Compito nel formato YYYY-MM-DD"
    )
    dettagli: Optional[str] = Field(
        None,
        description="Dettagli dell'assegnazione per Compito"
    )

    class Config:
        extra = 'forbid'  # Equivalent to "additionalProperties": false

class Attivita(BaseModel):
    attivita_id: Optional[int] = Field(
        None,
        description="ID unico dell'attività (generato dal database)"
    )
    tipo: str = Field(
        ...,
        description="Tipo di attività, ad esempio 'Interrogazione', 'Verifica', 'Compito', 'Argomento', 'Custom'"
    )
    descrizione: str = Field(
        ...,
        description="Descrizione dettagliata dell'attività"
    )
    data_scadenza: Optional[str] = Field(
        None,
        description="Data di scadenza nel formato YYYY-MM-DD"
    )
    priorita: int = Field(
        1,
        description="Priorità dell'attività: 1=Alta, 2=Media, 3=Bassa"
    )
    stato: bool = Field(
        False,
        description="Stato dell'attività: False=Non completata, True=Completata"
    )
    created_at: Optional[str] = Field(
        None,
        description="Data e ora di creazione dell'attività"
    )

    class Config:
        extra = 'forbid'  # Equivalent to "additionalProperties": false
