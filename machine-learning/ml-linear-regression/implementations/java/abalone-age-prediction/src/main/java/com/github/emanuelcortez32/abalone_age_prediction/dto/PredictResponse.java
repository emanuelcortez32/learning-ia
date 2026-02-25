package com.github.emanuelcortez32.abalone_age_prediction.dto;

public class PredictResponse {
    private float prediction;

    public PredictResponse(float prediction) {
        this.prediction = prediction;
    }

    public double getPrediction() {
        return prediction;
    }

    public void setPrediction(float prediction) {
        this.prediction = prediction;
    }
}
