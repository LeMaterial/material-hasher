from material_hasher.hashers.lehash import LeHasher

hashers = {
    c.__name__.lower(): c
    for c in [
        LeHasher,
    ]
}
"""
A dictionary of available hashers.

The keys are the names of the hashers (in lowercase) and the values are the corresponding hasher classes.
"""
