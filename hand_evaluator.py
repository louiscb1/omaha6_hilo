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
def evaluate_low_hand(cards):
    # Map rank characters to numerical values (Ace = 1)
    rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 1}

    # Extract the ranks of the 5 cards
    ranks = [rank_map[card[0]] for card in cards]

    # Filter for cards <= 8
    low_cards = sorted(set([r for r in ranks if r <= 8]))

    # If fewer than 5 unique low cards → no Low
    if len(low_cards) < 5:
        return None

    # Use the lowest 5 unique low cards
    five_low = low_cards[:5]

    # Lexicographical score: smaller is better
    # Example: A2345 → 102030405
    score = 0
    for r in five_low:
        score = score * 100 + r

    return score
