



from typing import Literal


ContentType = str | int | float | bool | None

StyleFormatType = Literal["title-format", "row-format", "list-format", "block-format", "list-layout"]

StyleProps = dict[StyleFormatType, str]
