import datetime as dt
from typing import List
from chatboard.model import Model, ModelField, KeyField, RelationField,Relation
from chatboard.model import NamespaceManager
from chatboard.model import Turn
from chatboard.model import ArtifactModel


class Like(ArtifactModel):
    post_id: int = ModelField(foreign_key=True)    

class Post(ArtifactModel):
    title: str = ModelField()
    content: str = ModelField()
    likes: Relation[Like] = RelationField(foreign_key="post_id")
    user_id: int = ModelField(foreign_key=True)


class User(Model):
    id: int = KeyField(primary_key=True)
    created_at: dt.datetime = ModelField(default_factory=dt.datetime.now)
    name: str = ModelField()
    age: int = ModelField()
    address: str = ModelField()
    posts: Relation[Post] = RelationField(foreign_key="user_id")

