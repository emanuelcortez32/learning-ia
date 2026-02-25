package com.github.emanuelcortez32.abalone_age_prediction.dto;

public class PredictRequest {
    private double sex;
    private double length;
    private double diameter;
    private double height;
    private double wholeWeight;
    private double shuckedWeight;
    private double visceraWeight;
    private double shellWeight;

    public PredictRequest(double sex, double length, double diameter, double height,
                          double wholeWeight, double shuckedWeight, double visceraWeight, double shellWeight) {
        this.sex = sex;
        this.length = length;
        this.diameter = diameter;
        this.height = height;
        this.wholeWeight = wholeWeight;
        this.shuckedWeight = shuckedWeight;
        this.visceraWeight = visceraWeight;
        this.shellWeight = shellWeight;
    }

    // Getters and Setters
    public double getSex() {
        return sex;
    }

    public void setSex(double sex) {
        this.sex = sex;
    }

    public double getLength() {
        return length;
    }

    public void setLength(double length) {
        this.length = length;
    }

    public double getDiameter() {
        return diameter;
    }

    public void setDiameter(double diameter) {
        this.diameter = diameter;
    }

    public double getHeight() {
        return height;
    }

    public void setHeight(double height) {
        this.height = height;
    }

    public double getWholeWeight() {
        return wholeWeight;
    }

    public void setWholeWeight(double wholeWeight) {
        this.wholeWeight = wholeWeight;
    }

    public double getShuckedWeight() {
        return shuckedWeight;
    }

    public void setShuckedWeight(double shuckedWeight) {
        this.shuckedWeight = shuckedWeight;
    }

    public double getVisceraWeight() {
        return visceraWeight;
    }

    public void setVisceraWeight(double visceraWeight) {
        this.visceraWeight = visceraWeight;
    }

    public double getShellWeight() {
        return shellWeight;
    }

    public void setShellWeight(double shellWeight) {
        this.shellWeight = shellWeight;
    }
}

