from app.chat.redis import client
import random


def random_component_by_score(component_type, component_map):
    # Ensure component type is 'llm', 'retriever', or 'memory'
    if component_type not in ["llm", "retriever", "memory"]:
        raise ValueError("Invalid component type")
    # From redis, get the hash containting the sum total of scores for each component type
    values = client.hgetall(f"{component_type}_score_values")
    # From redis, get the hash containing the number of times each component type has been scored
    counts = client.hgetall(f"{component_type}_score_counts")
    # Get all the valid component names from the component_map
    names = component_map.keys()
    # Loop over those valid names and calculate the average score for each component
    # Add average scores to a dictionary
    avg_scores = {}
    for name in names:
        print("got name", name)
        score = int(values.get(name, 1))
        count = int(counts.get(name, 1))
        avg = score / count
        avg_scores[name] = max(avg, 0.1)

    # Do a weighted random selection
    sum_scores = sum(avg_scores.values())
    random_val = random.uniform(0, sum_scores)
    cummulative = 0
    for name, score in avg_scores.items():
        cummulative += score
        if random_val <= cummulative:
            return name


def score_conversation(
    conversation_id: str, score: float, llm: str, retriever: str, memory: str
) -> None:
    """
    This function interfaces with langfuse to assign a score to a conversation, specified by its ID.
    It creates a new langfuse score utilizing the provided llm, retriever, and memory components.
    The details are encapsulated in JSON format and submitted along with the conversation_id and the score.

    :param conversation_id: The unique identifier for the conversation to be scored.
    :param score: The score assigned to the conversation.
    :param llm: The Language Model component information.
    :param retriever: The Retriever component information.
    :param memory: The Memory component information.

    Example Usage:

    score_conversation('abc123', 0.75, 'llm_info', 'retriever_info', 'memory_info')
    """
    # upvote = 1, downvote = 0
    score = min(max(score, 0), 1)

    client.hincrby("llm_score_values", llm, score)
    client.hincrby("llm_score_counts", llm, 1)

    client.hincrby("retriever_score_values", retriever, score)
    client.hincrby("retriever_score_values", retriever, 1)

    client.hincrby("memory_score_values", memory, score)
    client.hincrby("memory_score_values", memory, 1)


def get_scores():
    """
    Retrieves and organizes scores from the langfuse client for different component types and names.
    The scores are categorized and aggregated in a nested dictionary format where the outer key represents
    the component type and the inner key represents the component name, with each score listed in an array.

    The function accesses the langfuse client's score endpoint to obtain scores.
    If the score name cannot be parsed into JSON, it is skipped.

    :return: A dictionary organized by component type and name, containing arrays of scores.

    Example:

        {
            'llm': {
                'chatopenai-3.5-turbo': [score1, score2],
                'chatopenai-4': [score3, score4]
            },
            'retriever': { 'pinecone_store': [score5, score6] },
            'memory': { 'persist_memory': [score7, score8] }
        }
    """

    aggregate = {"llm": {}, "retriever": {}, "memory": {}}

    for component_type in aggregate.keys():
        values = client.hgetall(f"{component_type}_score_values")
        counts = client.hgetall(f"{component_type}_score_counts")

        names = values.keys()
        for name in names:
            score = int(values.get(name, 1))
            count = int(counts.get(name, 1))
            avg = score / count
            aggregate[component_type][name] = [avg]

    return aggregate
