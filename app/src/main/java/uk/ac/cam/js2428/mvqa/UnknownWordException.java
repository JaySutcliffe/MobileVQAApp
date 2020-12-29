package uk.ac.cam.js2428.mvqa;

public class UnknownWordException extends QuestionException {
    public UnknownWordException(String word) {
        super("The question asked uses a word untrained on:".concat(word));
    }
}
