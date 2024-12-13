def range_val(min, max):
    """Validates that an attribute's value is between two values, inclusive ([min, max])."""
    def validator(instance, attribute, value):
        if value < min or value > max:
            raise ValueError(f"{attribute} must be in range [{min}, {max}]")

    return validator