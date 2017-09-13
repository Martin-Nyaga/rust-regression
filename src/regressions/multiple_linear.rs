use utils::Dataset;
use regressions::{Regression, Linear};

#[derive(Debug)]
pub struct MultipleLinear {
    y: Dataset,
    xs: Vec<Dataset>,
    var_count: usize,
    pub coefficients: Vec<f64>,
    pub intercept: f64
}

#[derive(Debug)]
struct RankInfo {
    index: usize,
    used: bool,
    regression: Linear
}

impl MultipleLinear {
    pub fn new(y: Vec<f64>, xs: Vec<Vec<f64>>) -> MultipleLinear {
        let mut variables: Vec<RankInfo> = (0..xs.len())
            .map(|i| {
                let reg = Linear::new(y.clone(), xs[i].clone());
                RankInfo {
                    index: i,
                    used: false,
                    regression: reg 
                }
            })
            .collect();
        let mut intercept = 0.0;
        let mut coefficients: Vec<f64> = vec![0.0; xs.len()];

        while variables.iter().filter(|x| !x.used).collect::<Vec<_>>().len() > 0 {
            let minimising_variable_index = {
                let mut unused_variables: Vec<&RankInfo> = variables.iter()
                    .filter(|x| !x.used)
                    .collect();
                unused_variables.sort_by(|var1, var2| {
                    var1.regression.mean_square_error()
                        .partial_cmp(&var2.regression.mean_square_error())
                        .unwrap()
                });
                unused_variables[0].index
            };
            intercept = intercept + variables[minimising_variable_index].regression.intercept;
            coefficients[minimising_variable_index] = variables[minimising_variable_index].regression.gradient;
            coefficients[minimising_variable_index] = variables[minimising_variable_index].regression.gradient;
            variables[minimising_variable_index].used = true;
            let residuals = variables[minimising_variable_index].regression.residuals();
            for i in 0..variables.len() {
                variables[i].regression = Linear::new(residuals.clone(), xs[i].clone());
            }
        }

        MultipleLinear {
            y: Dataset::new(y),
            xs: xs.iter().map(|x| Dataset::new(x.to_owned())).collect(),
            var_count: xs.len(),
            coefficients,
            intercept
        }
    }

    fn x_data(&self) -> &Vec<Dataset> {
        &self.xs
    }

    fn predict_single(&self, xs: &Vec<f64>) -> f64 {
        let sum: f64 = xs.iter().zip(&self.coefficients).map(|(a,b)| a * b).sum();
        self.intercept + sum 
    }
    
    pub fn predictions(&self) -> Vec<f64> {
        let mut predictions = Vec::new();
        for i in 0..self.x_data()[0].len() - 1 {
            let arr: Vec<f64> = self.x_data()
                .iter()
                .map(|x_dataset| x_dataset.data[i])
                .collect(); 
            predictions.push(self.predict_single(&arr));
        }
        predictions
    }

    pub fn equation_string(&self) -> String {
        format!("{} {:?}", self.intercept, self.coefficients)
    }
}

#[cfg(test)]
mod multiple_linear_test {
    use super::*;

    #[test]
    fn it_generates_a_correct_multiple_linear_regression() {
    }
}
