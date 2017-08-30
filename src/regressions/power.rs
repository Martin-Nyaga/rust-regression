use utils::Dataset;
use regressions::{Regression, Linear};

// Form y = ax^b
// a := Coefficient
// b := Exponent
#[allow(dead_code)]
pub struct Power<'a> {
    x: Dataset<'a>,
    y: Dataset<'a>,
    pub coefficient: f64,
    pub exponent: f64
}

// Custom methods for this regression
impl<'a> Power<'a> {
    // Power regression is of the form y = ax^b
    // Convert to the log form ln(y) = b.ln(x) + ln(a) then
    // perform a standard linear regression on that new equation
    // and solve for b & a
    pub fn new(y: &'a Vec<f64>, x: &'a Vec<f64>) -> Power<'a> {
        let x_lns = x.iter()
            .map(|x1| x1.ln())
            .collect();

        let y_lns = y.iter()
            .map(|y1| y1.ln())
            .collect();

        let reg = Linear::new(&y_lns, &x_lns); 
        let exponent = reg.gradient;
        let coefficient = reg.intercept.exp();

        Power {
            x: Dataset::new(x),
            y: Dataset::new(y),
            coefficient,
            exponent
        }
    }
}

// Methods necessary to fulfil Regression trait 
impl<'a> Regression for Power<'a> {
    // Getter for x data for prediction purposes
    fn x_data(&self) -> &Vec<f64> {
        self.x.data
    }

    // Predict a y value from a single x value
    fn predict_single(&self, x: f64) -> f64 {
        self.coefficient * x.powf(self.exponent)
    }

    // Get equation string
    fn equation(&self) -> String {
        format!("y = {:.5}x^({:.5})", self.coefficient, self.exponent)
    }
}

#[cfg(test)]
mod power_tests {
    use super::*;

    #[test]
    fn it_generates_a_correct_power_regression_for_simple_data() {
        let x: Vec<f64> = (1u32..7u32).map(|x| x as f64).collect();
        // y = 2x^3
        let y: Vec<f64> = x.iter().map(|x| 2.0 * x.powf(3.0)).collect(); 
        let results = Power::new(&y, &x);

        assert_eq!(2.0, results.coefficient.round());
        assert_eq!(3.0, results.exponent.round());
    }
}

