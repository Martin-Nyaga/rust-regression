use utils::Dataset;
use regressions::{Regression, Linear};

// y = Ae^(bx)
// A = Coefficient
// b = Exponent
#[allow(dead_code)]
pub struct Exponential<'a> {
    x: Dataset<'a>,
    y: Dataset<'a>,
    pub coefficient: f64,
    pub exponent: f64
}

impl<'a> Exponential<'a> {
    // Exponential regression is of the form y = Ae^(bx)
    // solved by converting to the log form ln(y) = bx + lnA
    // and performing a simple linear regression to solve for b & A
    pub fn new(y: &'a Vec<f64>, x: &'a Vec<f64>) -> Exponential<'a> {
        let y_lns = y.iter()
            .map(|y1| y1.ln())
            .collect();

        let reg = Linear::new(&y_lns, x); 
        let exponent = reg.gradient;
        let coefficient = reg.intercept.exp();

        Exponential {
            x: Dataset::new(x),
            y: Dataset::new(y),
            coefficient,
            exponent
        }
    }
}

impl<'a> Regression for Exponential<'a> {
    fn x_data(&self) -> &Dataset {
        &self.x
    }

    fn y_data(&self) -> &Dataset {
        &self.y
    }

    fn predict_single(&self, x: f64) -> f64 {
        self.coefficient * (self.exponent * x).exp()
    }

    fn equation_string(&self) -> String {
        format!("y = {:.5}e^({:.5}x)", self.coefficient, self.exponent)
    }
}

#[cfg(test)]
mod exponential_tests {
    use super::*;

    #[test]
    fn it_generates_a_correct_exponential_regression_for_simple_data() {
        let x: Vec<f64> = (1u32..7u32).map(|x| x as f64).collect();
        // y = 2e^(3x)
        let y: Vec<f64> = x.iter().map(|x| 2.0 * (3.0*x).exp()).collect(); 
        let results = Exponential::new(&y, &x);

        assert_eq!(2.0, results.coefficient);
        assert_eq!(3.0, results.exponent);
    }
}

