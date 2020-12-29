package uk.ac.cam.js2428.mvqa;

public class UnknownAnswerException extends QuestionException {
    public UnknownAnswerException(int answer) {
        super("Answer returned is not present in the lookup table: " + answer);
    }
}
