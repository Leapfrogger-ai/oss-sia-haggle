# Qualitative: where the base Llama-3.3-70B fails (base prediction vs gold)

Base model accuracy on the 60-item held-out set: **53.3%** — it is wrong on **28** cases.
These are the cases the fine-tune targets. The LoRA's held-out loss drop (0.69→0.375) is
the model learning to assign higher probability to exactly these gold answers.

_Note: this compares base-vs-gold. A base-vs-tuned comparison needs the tuned model served (gated)._

---

### IKEA standing desk w/ adjustable legs $60 OBO  (furniture)
- List $60.0 | buyer target $45.0 | seller target $60.0
- **Base model predicted:** `NO_DEAL`   →   **Truth:** `15.0`  ❌
- Hidden closing turns (what the model had to infer):
    > seller: Deal.
    > buyer: Great! Thanks!

### Aeron Chairs Size C  (furniture)
- List $299.0 | buyer target $275.0 | seller target $299.0
- **Base model predicted:** `NO_DEAL`   →   **Truth:** `250.0`  ❌
- Hidden closing turns (what the model had to infer):
    > seller: Well I think it woudl work for you but that is your choice, I am looking to get rid of it because I am moving. I COULD COME DOWN TO 250 AND FREE DELIV
    > buyer: Deal!

### "Reduced" Custom made reclaimed wood console  (furniture)
- List $500.0 | buyer target $300.0 | seller target $500.0
- **Base model predicted:** `400.0`   →   **Truth:** `475.0`  ❌
- Hidden closing turns (what the model had to infer):
    > seller: I custom make these and the materials are quite expensive and they are labor intensive. The most I could offer is a 5% discount which would take it to
    > buyer: Ok,I can do that

### Highly Sought After Dublin Ranch Townhouse 2BD/2BA/2 Car Gar  (housing)
- List $2875.0 | buyer target $2012.0 | seller target $2875.0
- **Base model predicted:** `2450.0`   →   **Truth:** `2100.0`  ❌
- Hidden closing turns (what the model had to infer):
    > seller: I could pay for the sewer and trash removal if you willing to accept my offer of $2100 in I will pay your first light bill!
    > buyer: That sounds like a fantastic deal! Thanks for your flexibility!

### 2013 Hyundai Elantra GLS 4dr Sedan Gas SAVER  (car)
- List $10500.0 | buyer target $9450.0 | seller target $10500.0
- **Base model predicted:** `10000.0`   →   **Truth:** `NO_DEAL`  ❌
- Hidden closing turns (what the model had to infer):
    > buyer: Well, what are the problems with the car?
    > seller: There is no problems with it whatsover. It has 84k miles though.
