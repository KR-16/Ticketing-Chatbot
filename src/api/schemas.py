"""Request/response models for the inference API."""

from pydantic import BaseModel, Field, model_validator


class TicketRequest(BaseModel):
    subject: str = Field(
        default="",
        max_length=1000,
        description="Ticket subject line",
        examples=["Data analytics platform crashes on startup"],
    )
    body: str = Field(
        default="",
        max_length=20000,
        description="Ticket body text",
        examples=[
            "The platform shut down unexpectedly due to low memory. "
            "Restarting did not help. I need assistance resolving this."
        ],
    )

    @model_validator(mode="after")
    def require_some_text(self):
        if not (self.subject.strip() or self.body.strip()):
            raise ValueError("Provide a non-empty subject or body")
        return self


class PredictionResponse(BaseModel):
    predicted_type: str = Field(examples=["Incident"])
    confidence: float = Field(ge=0.0, le=1.0, examples=[0.973])
    probabilities: dict[str, float] = Field(
        description="Probability per ticket type, summing to ~1.0",
        examples=[{
            "Change": 0.004, "Incident": 0.973,
            "Problem": 0.019, "Request": 0.004,
        }],
    )


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    classes: list[str] = Field(
        default_factory=list,
        examples=[["Change", "Incident", "Problem", "Request"]],
    )
