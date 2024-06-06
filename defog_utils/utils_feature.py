from dataclasses import asdict, dataclass


@dataclass
class Features:
    """
    Base class for feature classes implementing a few standard helper functions
    for serialization, conversion and display of features.
    We allow for a `_prefix` to be added to the feature names to disambiguate
    features from different classes (e.g. num_tables)
    To avoid adding the _prefix alongside the feature names, we skip any fields
    starting with "_". All accesses to get the features (names/values) should
    be done through the `.to_dict()` method to avoid getting the `_prefix`.
    """

    _prefix: str = ""

    def feature_names(self):
        """
        Return a list of feature names, prefixed if necessary.
        """
        return list(self.to_dict().keys())

    def num_features(self):
        """
        Return the number of features.
        """
        return len(self.to_dict())

    def to_dict(self):
        """
        Convert the dataclass into a dictionary.
        """
        # remove _prefix as it's only used for getting feature names for display
        d = {}
        for k, v in asdict(self).items():
            if not k.startswith("_"):
                if self._prefix:
                    d[f"{self._prefix}_{k}"] = v
                else:
                    d[k] = v
        return d

    def compact(self):
        """
        Serialize all values as a comma-separated string with 0 for False and 1 for True.
        Preserves the order as listed/defined above, and skips any fields starting with "_".
        """
        values = [str(int(v)) for v in self.to_dict().values()]
        return ",".join(values)

    def from_compact(self, compact_str):
        """
        Initialize the dataclass from a compact string.
        """
        values = []
        str_split = compact_str.split(",")
        if len(str_split) != self.num_features():
            raise ValueError(
                f"Compact string has {len(str_split)} values, expected {self.num_features()}"
            )
        # get members of class starting with _ and append default value for the class to values
        values.append(self._prefix)
        for v in str_split:
            try:
                values.append(int(v))
            except ValueError as e:
                print(f"Error parsing value {v}: {e}")
                values.append(0)
        return self.__class__(*values)

    def positive_features(self):
        """
        Return a list of features that are not False or 0 as a dictionary.
        """
        return {k: v for k, v in self.to_dict().items() if v}
