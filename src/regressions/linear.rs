use utils::Dataset;
use regressions::Regression;

#[derive(Debug)]
pub struct Linear {
    x: Dataset,
    y: Dataset,
    pub gradient: f64,
    pub intercept: f64,
    pub covariance: f64
}

impl Linear {
    pub fn new(y: Vec<f64>, x: Vec<f64>) -> Linear {
        let mut x = Dataset::new(x);
        let mut y = Dataset::new(y);
        
        let covariance = Linear::covariance(&mut x, &mut y); 
        let gradient = covariance / x.variance();
        let intercept = y.mean() - (gradient * x.mean());

        Linear {
            x,
            y,
            gradient,
            intercept,
            covariance
        }
    }

    fn covariance(mut x: &mut Dataset, mut y: &mut Dataset) -> f64 {
        let products: Vec<f64> = x.diffs()
            .iter()
            .zip(y.diffs())
            .map(|(x1, y1)| x1 * y1)
            .collect();
        let sum: f64 = products.iter().sum();
        sum / (x.data.len() as f64 - 1.0)
    }

    pub fn pearsons_correlation(&mut self) -> f64 {
        self.covariance / (self.x.stdev() * self.y.stdev())
    }
}

impl Regression for Linear {
    fn x_data(&self) -> &Dataset {
        &self.x
    }

    fn y_data(&self) -> &Dataset {
        &self.y
    }
    
    fn predict_single(&self, x: f64) -> f64 {
        (self.gradient * x) + self.intercept
    }

    fn equation_string(&self) -> String {
        format!("y = {:.5}x + {:.5}", self.gradient, self.intercept)
    }
}

#[cfg(test)]
mod linear_tests {
    use super::*;

    #[test]
    fn it_generates_a_correct_linear_regression_for_simple_data() {
        let x: Vec<f64> = (1u32..7u32).map(|x| x as f64).collect();
        // y = 3x + 4
        let y: Vec<f64> = x.iter().map(|x| 3.0*x + 4.0).collect(); 
        let results = Linear::new(&y, &x);

        assert_eq!(3.0, results.gradient);
        assert_eq!(4.0, results.intercept);
    }
}

