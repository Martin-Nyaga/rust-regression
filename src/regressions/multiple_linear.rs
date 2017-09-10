use utils::Dataset;
use regressions::{Regression, Linear};

#[derive(Debug)]
pub struct MultipleLinear<'a> {
    y: Dataset<'a>,
    xs: Vec<Dataset<'a>>,
    var_count: usize,
    pub coefficients: Vec<f64>,
    pub intercept: f64
}

struct RankInfo<'a> {
    index: usize,
    used: bool,
    regression: Linear<'a>
}

impl<'a> MultipleLinear<'a> {
    pub fn new(y: &'a Vec<f64>, xs: &'a Vec<&'a Vec<f64>>) -> MultipleLinear<'a> {
        let mut variables: Vec<RankInfo> = (0..xs.len())
            .map(|i| {
                RankInfo {
                    index: i,
                    used: false,
                    regression: Linear::new(y, &xs[i])
                }
            })
            .collect();
        let mut intercept = 0.0;
        let mut coefficients: Vec<f64> = vec![0.0; xs.len()];

        while variables.iter().filter(|x| !x.used).collect::<Vec<&RankInfo>>().len() > 0 {
            let minimising_variable_index = {
                let mut unused_variables: Vec<&RankInfo> = variables.iter()
                    .filter(|x| !x.used)
                    .collect();
                unused_variables.sort_by(|var1, var2| {
                    var2.regression.mean_square_error().partial_cmp(&var1.regression.mean_square_error()).unwrap()
                });
                let mut minimising_variable = unused_variables[0];
                intercept = intercept + minimising_variable.regression.intercept;
                coefficients[minimising_variable.index] = minimising_variable.regression.gradient;
                coefficients[minimising_variable.index] = minimising_variable.regression.gradient;
                minimising_variable.index
            };
            variables[minimising_variable_index].used = true;
        }

        MultipleLinear {
            y: Dataset::new(y),
            xs: xs.iter().map(|x| Dataset::new(x)).collect(),
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
        format!("{:?}", self.coefficients)
    }
}

#[cfg(test)]
mod multiple_linear_test {
    use super::*;

    #[test]
    fn it_generates_a_correct_exponential_regression_for_simple_data() {
        // TODO: Test multiple linear regression
    }
}
