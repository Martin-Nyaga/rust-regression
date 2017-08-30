use utils::Dataset;
use regressions::Linear;

// Form y = Ae^(bx)
// A := Coefficient
// b := Exponent
pub struct Exponential<'a> {
    x: Dataset<'a>,
    y: Dataset<'a>,
    pub coefficient: f64,
    pub exponent: f64
}

impl<'a> Exponential<'a> {
    pub fn new(x: &'a Vec<f64>, y: &'a Vec<f64>) -> Exponential<'a> {
        let x_lns = x.iter()
            .map(|x| x.ln())
            .collect();
    
        let y_lns = y.iter()
            .map(|y| y.ln())
            .collect();

        let reg = Linear::new(&x_lns, &y_lns); 
        let exponent = reg.gradient;
        let coefficient = reg.intercept.exp();

        Exponential {
            x: Dataset::new(x),
            y: Dataset::new(y),
            coefficient,
            exponent
        }
    }

    pub fn predictions(&self) -> Vec<f64> {
        self.x.data
            .iter()
            .map(|x| self.coefficient * (self.exponent * x).exp())
            .collect()
    }
}


#[cfg(test)]
mod exponential {
    use super::*;

    #[test]
    fn it_generates_a_correct_linear_regression_for_simple_data() {
        let x: Vec<f64> = (1u32..7u32).map(|x| x as f64).collect();
        // y = 2e^(3x)
        let y: Vec<f64> = x.iter().map(|x| 2.0 * (3.0*x).exp()).collect(); 
        let results = Exponential::new(&x, &y);

        assert_eq!(2.0, results.coefficient);
        assert_eq!(3.0, results.exponent);
    }
}

