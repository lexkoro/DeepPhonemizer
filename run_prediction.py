from dp.phonemizer import Phonemizer

if __name__ == "__main__":

    checkpoint_path = "checkpoints/best_model_no_optim.pt"
    phonemizer = Phonemizer.from_checkpoint(checkpoint_path)

    sentences = [
        "Aus einem fernen Land, von dem du wahrscheinlich noch nie etwas gehört hast, Marvin.",
        "Moment, woher kennst du meinen Namen?",
        "Ich bin eine gute Zuhörerin und hier unter Deck ist es nicht gerade geräumig.",
        "Du scheinst ein netter Junge zu sein. Ich habe ein Angebot für dich.",
        "Siehst du den Mistkerl, der die Pfeile schnitzt? Dieses Arschloch hat meine Schatulle genommen und sie bei seiner Beute versteckt. Ich habe gesehen, wie er damit im Frachtraum verschwunden ist.",
        "Ich dachte, er würde meine Schatulle in einer Truhe verstecken, aber dann konnte ich das laute Knarzen von Schiffsplanken hören. Könntest du dich im Frachtraum mal umsehen und schauen, ob es dort ein Versteck gibt?",
        "Ich habe deine Schatulle gefunden.",
        "Das ist ja grossartig! Vielen Dank, mein Junge!",
        "Im Moment habe ich leider keine Möglichkeit dich für deine Hilfe zu bezahlen. Aber ich verspreche dir, sobald wir auf Archolos sind sorge ich dafür, dass meine Freunde dich entlohnen!"
        "Sieht aus, als wären wir bald da.",
        "Woher willst du das wissen?",
        "WAS SOLL DAS?",
        "Das spiel heisst Gothic.",
    ]

    for text in sentences:
        result = phonemizer.phonemise_list([text], lang="de")
        print(result.predictions)
        print(result.phonemes)

    # for text, pred in result.predictions.items():
    #     tokens, probs = pred.phoneme_tokens, pred.token_probs
    #     for o, p in zip(tokens, probs):
    #         print(f"{o} {p}")
    #     tokens = "".join(tokens)
    #     print(f"{text} | {tokens} | {pred.confidence}")
