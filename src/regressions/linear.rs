use utils::Dataset;

#[derive(Debug)]
pub struct Linear<'a> {
    x: Dataset<'a>,
    y: Dataset<'a>,
    pub gradient: f64,
    pub intercept: f64,
    pub covariance: f64
}

impl<'a> Linear<'a> {
    pub fn new(x: &'a Vec<f64>, y: &'a Vec<f64>) -> Linear<'a> {
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

    pub fn predictions(&self) -> Vec<f64> {
        self.x.data
            .iter()
            .map(|x| (self.gradient * x) + self.intercept)
            .collect()
    }

    pub fn predict_single(&self, x: f64) -> f64 {
        (self.gradient * x) + self.intercept
    }

    pub fn predict_multi(&self, x: &Vec<f64>) -> Vec<f64> {
        x.iter()
            .map(|x| (self.gradient * x) + self.intercept)
            .collect()
    }
}

#[cfg(test)]
mod linear {
    use super::*;

    #[test]
    fn it_generates_a_correct_linear_regression_for_simple_data() {
        let x: Vec<f64> = (1u32..7u32).map(|x| x as f64).collect();
        // y = 3x + 4
        let y: Vec<f64> = x.iter().map(|x| 3.0*x + 4.0).collect(); 
        let results = Linear::new(&x, &y);

        assert_eq!(3.0, results.gradient);
        assert_eq!(4.0, results.intercept);
    }
}

