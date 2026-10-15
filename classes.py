"""CLASS_SETS holds the label sets every script in this repo selects from with `--set`. Each set is
10 Banking77 intent names, each with the one-sentence definition that is the only description
generate.py gives the LLM.

`distinct` is 10 intents on unrelated topics. `confusable` is 5 pairs of intents on the same topic,
listed pair by pair; each definition states what separates the intent from the other one in its
pair."""

CLASS_SETS = {
    "distinct": {
        "card_arrival":        "The customer ordered a card that has not arrived yet and asks where it is.",
        "lost_or_stolen_card": "The customer reports that their card has been lost or stolen.",
        "exchange_rate":       "The customer asks what exchange rate is applied when converting currencies.",
        "change_pin":          "The customer wants to change their card's PIN and asks how or where to do it.",
        "terminate_account":   "The customer wants to close their account.",
        "age_limit":           "The customer asks how old someone must be to open an account.",
        "atm_support":         "The customer asks which ATMs accept their card or where they can withdraw cash.",
        "apple_pay_or_google_pay": "The customer asks about using or setting up Apple Pay or Google Pay.",
        "request_refund":        "The customer wants a refund for something they bought and asks how to get it.",
        "edit_personal_details": "The customer wants to update their name, address or phone number.",
    },
    "confusable": {
        "card_arrival": (
            "The customer has already waited for a card they ordered, it has not come, and they ask where it "
            "is — a complaint about a card that is late or missing, not a question about normal delivery "
            "times."
        ),
        "card_delivery_estimate": (
            "The customer asks how long card delivery takes or when a card will arrive, as a question about "
            "timing or delivery options — they are not complaining that a card is late or missing."
        ),
        "declined_transfer": (
            "The customer tried to send a transfer and it was declined or rejected at the moment they "
            "attempted it, and they ask why — the transfer was refused up front, not sent and then failed."
        ),
        "failed_transfer": (
            "The customer's transfer did not go through or came back as failed after they had sent it, and "
            "they ask what went wrong — the transfer was attempted and failed, not refused up front as "
            "declined."
        ),
        "card_payment_wrong_exchange_rate": (
            "The customer paid with their card in a shop or online in a foreign currency and says the "
            "exchange rate applied was wrong — a card purchase, not a cash withdrawal."
        ),
        "wrong_exchange_rate_for_cash_withdrawal": (
            "The customer withdrew cash from an ATM in a foreign currency and says the exchange rate applied "
            "was wrong — a cash withdrawal, not a card purchase."
        ),
        "getting_virtual_card": (
            "The customer asks how to get or where to find a virtual card, the ordinary one that is kept and "
            "reused — not a disposable card that expires after one use."
        ),
        "get_disposable_virtual_card": (
            "The customer asks how to get a disposable virtual card, the single-use kind that expires after "
            "one payment — not the ordinary virtual card that is kept and reused."
        ),
        "verify_my_identity": (
            "The customer asks how to verify their identity or what documents and steps are needed — they "
            "accept that it must be done and want to know how, not why."
        ),
        "why_verify_identity": (
            "The customer asks why they have to verify their identity at all, or objects to being asked — "
            "they want the reason, not the steps."
        ),
    },
}
