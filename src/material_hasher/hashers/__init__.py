from material_hasher.hashers.lehash import LeHasher

hashers = [LeHasher]
hashers = {c.__name__.lower(): c for c in hashers}
