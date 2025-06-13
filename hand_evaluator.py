import random
from treys import Card, Evaluator

# Initialize the evaluator
evaluator = Evaluator()

# Helper: convert list of string cards to treys Card ints
def convert_cards_to_treys(cards):
    return [Card.new(card) for card in cards]

# Returns the "score" for High hand (lower is better in treys, we negate it)
def evaluate_high_hand(cards):
    treys_cards = convert_cards_to_treys(cards)
    board = treys_cards[2:]  # 3 board cards
    hand = treys_cards[:2]   # 2 hole cards
    score = evaluator.evaluate(board, hand)
    return -score  # We negate it so HIGHER is better in our simulator!

# Returns Low hand score or None if no Low
def evaluate_low_hand(cards, board_cards):
    # First check board — must have at least 3 distinct low ranks (8 or lower)
    board_low_ranks = set(card[0] for card in board_cards if card[0] in "2345678")
    if len(board_low_ranks) < 3:
        return None  # No low possible on this board
    
    # Now check player hand + board combo
    low_cards = [card for card in cards if card[0] in "2345678"]
    low_ranks = sorted(set(card[0] for card in low_cards), key=lambda x: "A2345678".index(x))
    
    if len(low_ranks) >= 5:
        # Return numeric score of low — e.g. A2345 = 12345, 23456 = 23456, etc
        rank_to_value = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8}
        score = int("".join(str(rank_to_value[r]) for r in low_ranks[:5]))
        return score
    else:
        return None

