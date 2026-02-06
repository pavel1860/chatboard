from chatboard import view, prompt
from pydantic import BaseModel, Field  


class UserStats(BaseModel):
    name: str = Field("unknown", title="Name", description="The name of the player")
    health: int = Field(..., title="Health", description="The health of the player", min=0, max=100)
    mana: int = Field(..., title="Mana", description="The mana of the player", min=0, max=100)
    strength: int = Field(..., title="Strength", description="The strength of the player", min=0, max=10)
    reputation: int = Field(..., title="Reputation", description="The reputation of the player", min=0, max=100)
    
class PirateStats(BaseModel):
    name: str = Field("Jack Sparrow", title="Name", description="your name")
    health: int = Field(..., title="Health", description="your health", min=0, max=100)
    mana: int = Field(..., title="Mana", description="your mana", min=0, max=100)
    strength: int = Field(..., title="Strength", description="your strength", min=0, max=10)
    reputation: int = Field(..., title="Reputation", description="your reputation", min=0, max=100)
    


class AttackAction(BaseModel):
    """use this action when you want to attack the enemy"""
    damage: int = Field(..., title="Damage", description="The amount of damage the attack does", min=0, max=100)
    mana_cost: int = Field(..., title="Mana Cost", description="The amount of mana required to perform the attack", min=0, max=100)

class GiveQuestAction(BaseModel):
    """use this action when you want to give the player a quest"""
    description: str = Field(..., title="Description", description="The description of the quest")
    task: str = Field(..., title="Task", description="The task the player needs to complete")
    reward: int = Field(..., title="Reward", description="The reward for completing the quest", min=0)
    
class ChangeReputationAction(BaseModel):
    """use this action when you want to change the player's reputation. it can be positive if you like the player or negative if you don't"""
    reputation: int = Field(..., title="Reputation", description="The reputation of the player", min=0, max=100)
    
class MoveAction(BaseModel):
    """you should use this if you want to move to a different location"""
    direction: str = Field(..., title="Direction", description="The direction the player wants to move in. could be 'north', 'south', 'east', 'west'")
    distance: int = Field(..., title="Distance", description="The distance the player wants to move", min=0)
    



@view(title="User Stats")
def user_stats_view(stats: UserStats):
    return stats


@view(title="Pirate Stats")
def pirate_stats_view(stats: PirateStats):
    return stats

@view()
def user_message_view(message):
    return message

@view(
    role="system",    
)
def pirate_view(pirate_stats: PirateStats, user_stats: UserStats):
    return pirate_stats_view(pirate_stats), user_stats_view(user_stats)

# @prompt(
#     model="gpt-4o",
#     background="you are a pirate by the name of Jack Sparrow.",  
#     rules=[
#         "you can answer with a regular message or use an action.",
#         # "if you use an action, you must create a chain of thought, step by step reasoning for that action.",
#         "you if you provide a message, it should be without formatting. you should answer as if you are speaking.",
#     ],
#     actions=[AttackAction, GiveQuestAction, MoveAction, ChangeReputationAction]
# )
# def pirate_prompt(message, pirate_stats, user_stats):
#     return pirate_view(pirate_stats, user_stats), user_message_view(message)


# pirate_stats = PirateStats(health=100, mana=30, strength=8, reputation=50)
# user_stats = UserStats(health=100, mana=50, strength=10, reputation=50)

# message = "Hi there matey, give me a quest"
# response = await pirate_prompt(
#     message=message,
#     pirate_stats=pirate_stats,
#     user_stats=user_stats
# )
# response




from chatboard import AgentRouter
from chatboard.agent.action_agent import ActionAgent
from chatboard import ChatPrompt

# pirate_agent = AgentRouter(
#         name="pirate_agent", 
#         prompt_cls=ChatPrompt, 
#         add_input_history=True,
#         iterations=2
#     )
pirate_agent = ActionAgent(
        name="pirate_agent", 
        prompt_cls=ChatPrompt, 
        add_input_history=True,
        iterations=2
    )



@pirate_agent.prompt(
    # model="gpt-4o",
    model="claude-3-5-sonnet-20240620",
    background="you are a pirate by the name of Jack Sparrow.",  
    rules=[
        "you can answer with a regular message or use an action.",
        # "if you use an action, you must create a chain of thought, step by step reasoning for that action.",
        "you if you provide a message, it should be without formatting. you should answer as if you are speaking.",
    ],
    actions=[AttackAction, GiveQuestAction, MoveAction, ChangeReputationAction],
    # tool_choice="required"
)
def pirate_prompt(context, message, pirate_stats, user_stats):
    if message.role == 'tool':
        return [
            *context.history.get(10),
            message
        ]
        
    return [
        *context.history.get(10),
        pirate_view(pirate_stats, user_stats), 
        message
        # user_message_view(message)
    ]

@pirate_agent.reducer(action=AttackAction)
def attack_reducer(action, pirate_stats, user_stats):
    pirate_stats.health -= action.damage
    pirate_stats.mana -= action.mana_cost
    return pirate_stats

@pirate_agent.reducer(action=GiveQuestAction)
def give_quest_reducer(action, pirate_stats, user_stats):
    user_stats.reputation += 10
    return pirate_stats

@pirate_agent.reducer(action=MoveAction)
def move_reducer(action, pirate_stats, user_stats):
    return pirate_stats

@pirate_agent.reducer(action=ChangeReputationAction)
def change_reputation_reducer(action, pirate_stats, user_stats):
    user_stats.reputation += action.reputation
    return pirate_stats



