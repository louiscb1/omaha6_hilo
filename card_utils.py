import random

SUITS = ['c', 'd', 'h', 's']
RANKS = '23456789TJQKA'

# Create full deck
def create_deck():
    return [rank + suit for rank in RANKS for suit in SUITS]

# Deal n cards
def deal_cards(deck, n):
    return [deck.pop() for _ in range(n)]

# Shuffle deck
def shuffle_deck(deck):
    random.shuffle(deck)
