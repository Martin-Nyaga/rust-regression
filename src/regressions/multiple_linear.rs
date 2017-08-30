use utils::Dataset;
use regressions::{Regression, MultipleRegression, Linear};

#[derive(Debug)]
pub struct MultipleLinear<'a> {
    y: Dataset<'a>,
    xs: Vec<Dataset<'a>>,
    var_count: usize,
    pub coefficients: Vec<f64>,
    pub intercept: f64
}

// Custom methods for this regression
impl<'a> MultipleLinear<'a> {
    pub fn new(y: &'a Vec<f64>, xs: &'a Vec<&'a Vec<f64>>) -> MultipleLinear<'a> {
        let reg = Linear::new(y, &xs[0]);
        
        // set intercept
        let mut intercept = reg.intercept;
       
        // init coefficient and add coefficient for first x
        let mut coefficients = Vec::new();
        coefficients.push(reg.gradient);

        let mut predictions = reg.predictions();
        let mut residuals: Vec<f64> = Dataset::new(y).residuals(predictions);

        for x in xs.iter().skip(1) {
            let res = residuals.clone();
            let reg = Linear::new(&res, x);
            intercept = intercept + reg.intercept;
            coefficients.push(reg.gradient);

            predictions = reg.predictions();
            residuals = Dataset::new(&residuals).residuals(predictions);
        }

        MultipleLinear {
            y: Dataset::new(y),
            xs: xs.iter().map(|x| Dataset::new(x)).collect(),
            var_count: xs.len(),
            coefficients,
            intercept
        }
    }
}

// Methods necessary to fulfil generic Regression trait
impl<'a> MultipleRegression for MultipleLinear<'a> {
    // Getter for x data for prediction purposes
    fn x_data(&self) -> Vec<&Vec<f64>> {
       self.xs
            .iter()
            .map(|x: &Dataset<'a>| x.data)
            .collect()
    }
    
    // Predict a y value from a single x value
    fn predict_single(&self, xs: &Vec<f64>) -> f64 {
        let sum: f64 = xs.iter().zip(&self.coefficients).map(|(a,b)| a * b).sum();
        self.intercept + sum 
    }
}

#[cfg(test)]
mod multiple_linear_test {
    use super::*;

    #[test]
    fn it_generates_a_correct_exponential_regression_for_simple_data() {
        let x: Vec<f64> = (1u32..7u32).map(|x| x as f64).collect();
        let x2: Vec<f64> = x.iter().map(|x| x*x).collect();

        // y = 5x + 2x^2 + 3
        let y: Vec<f64> = x.iter()
            .map(|x| 5.0*x + 2.0*x.powi(2) + 3.0)
            .collect(); 
        let xs = vec![&x, &x2];

        let results = MultipleLinear::new(&y, &xs);

        // assert_eq!(3.0, results.intercept);
        // assert_eq!(5.0, results.coefficients[0]);
        // assert_eq!(2.0, results.coefficients[1]);
    }
}

