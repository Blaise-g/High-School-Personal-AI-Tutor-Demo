import math
from datetime import datetime, timedelta

class SpacedRepetition:
    def __init__(self):
        self.flashcards = {}

    def add_flashcard(self, card_id, question, answer):
        self.flashcards[card_id] = {
            'question': question,
            'answer': answer,
            'easiness': 2.5,
            'interval': 0,
            'repetitions': 0,
            'next_review': datetime.now()
        }
        print(f"Flashcard added: {self.flashcards[card_id]}")

    def review_flashcard(self, card_id, quality):
        if card_id not in self.flashcards:
            print(f"Flashcard ID not found: {card_id}")
            print(f"Available flashcards: {list(self.flashcards.keys())}")
            raise ValueError("Flashcard not found")

        card = self.flashcards[card_id]
        card['easiness'] = max(1.3, card['easiness'] + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        card['repetitions'] += 1

        if quality < 3:
            card['interval'] = 0
        elif card['repetitions'] == 1:
            card['interval'] = 1
        elif card['repetitions'] == 2:
            card['interval'] = 6
        else:
            card['interval'] = round(card['interval'] * card['easiness'])

        card['next_review'] = datetime.now() + timedelta(days=card['interval'])

    def get_due_flashcards(self):
        now = datetime.now()
        return [card_id for card_id, card in self.flashcards.items() if card['next_review'] <= now]

    def get_flashcard(self, card_id):
        return self.flashcards.get(card_id)

spaced_repetition = SpacedRepetition()
