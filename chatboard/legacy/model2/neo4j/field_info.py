from chatboard.model.base_namespace import NSFieldInfo

class Neo4jFieldInfo(NSFieldInfo):
    """
    FieldInfo for Neo4j node properties.
    Handles serialization and deserialization of Python values to Neo4j-compatible types.
    """
    def serialize(self, value):
        # For Neo4j, most simple types (str, int, float, bool, list) are supported.
        # For enums/UUIDs/datetime, convert to str as needed.
        if value is None:
            return None
        if self.is_enum:
            return value.value if hasattr(value, "value") else str(value)
        # Neo4j supports datetime (in recent versions), but for compatibility, you may want to store as ISO string:
        if hasattr(value, "isoformat"):
            return value.isoformat()
        # Add more custom type handling as needed
        return value

    def deserialize(self, value):
        # Reverse of serialize; e.g., parse datetime string back to datetime
        if value is None:
            return None
        if self.is_enum and self.enum_values:
            # Assume enum type; look up value
            for enum_val in self.enum_values:
                if value == enum_val:
                    return self.data_type(enum_val)
        # Could handle datetime/UUID/etc. here
        return value
