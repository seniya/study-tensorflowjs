import * as tf from "@tensorflow/tfjs";

export interface CarType {
  Name: string;
  Miles_per_Gallon: number;
  Cylinders: number;
  Displacement: number;
  Horsepower: number;
  Weight_in_lbs: number;
  Acceleration: number;
  Year: string;
  Origin: string;
}

export interface Car2DType {
  mpg: number;
  horsepower: number;
}

export interface NormalizationType {
  inputs: tf.Tensor<tf.Rank>;
  labels: tf.Tensor<tf.Rank>;
  inputMax: tf.Tensor<tf.Rank>;
  inputMin: tf.Tensor<tf.Rank>;
  labelMax: tf.Tensor<tf.Rank>;
  labelMin: tf.Tensor<tf.Rank>;
}
