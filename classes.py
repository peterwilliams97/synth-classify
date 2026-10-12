"""CLASSES is the label set every script in this repo shares: 10 Banking77 intent names, each with
the one-sentence definition that is the only description generate.py gives the LLM."""

CLASSES = {
    "card_arrival": "The customer ordered a card that has not arrived yet and asks where it is.",
    "lost_or_stolen_card": "The customer reports that their card has been lost or stolen.",
    "exchange_rate": "The customer asks what exchange rate is applied when converting currencies.",
    "change_pin": "The customer wants to change their card's PIN and asks how or where to do it.",
    "terminate_account": "The customer wants to close their account.",
    "age_limit": "The customer asks how old someone must be to open an account.",
    "atm_support": "The customer asks which ATMs accept their card or where they can withdraw cash.",
    "apple_pay_or_google_pay": "The customer asks about using or setting up Apple Pay or Google Pay.",
    "request_refund": "The customer wants a refund for something they bought and asks how to get it.",
    "edit_personal_details": "The customer wants to update their name, address or phone number.",
}
