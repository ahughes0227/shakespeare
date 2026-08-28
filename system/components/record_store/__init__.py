"""The record_store family.

Persist one row per item, and read the table back, so that what a model reads survives
outside the response that reported it.

Its contract, risks and failure modes are declared once in `family-context.yml` and
inherited by every operator here.
"""
