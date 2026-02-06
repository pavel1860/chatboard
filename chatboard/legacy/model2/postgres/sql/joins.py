





class Join:
    def __init__(self, table, condition, join_type='INNER', alias=None):
        self.table = table        # Table object or string
        self.condition = condition  # Expression object
        self.join_type = join_type.upper()
        self.alias = alias

    def __repr__(self):
        return f"<{self.join_type} JOIN {self.table} ON {self.condition}>"





class InnerJoin(Join):
    def __init__(self, table, condition, alias=None):
        super().__init__(table, condition, 'INNER', alias)

class LeftJoin(Join):
    def __init__(self, table, condition, alias=None):
        super().__init__(table, condition, 'LEFT', alias)

class RightJoin(Join):
    def __init__(self, table, condition, alias=None):
        super().__init__(table, condition, 'RIGHT', alias)

class FullJoin(Join):
    def __init__(self, table, condition, alias=None):
        super().__init__(table, condition, 'FULL', alias)
