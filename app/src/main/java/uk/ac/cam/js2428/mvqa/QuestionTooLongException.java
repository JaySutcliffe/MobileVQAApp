package uk.ac.cam.js2428.mvqa;

public class QuestionTooLongException extends QuestionException{
    public QuestionTooLongException(int size) {
        super("The question entered exceeds a maximum size of " + size);
    }
}
