#[derive(Debug)]
pub struct Dataset {
    pub data: Vec<f64>,
    mean: Option<f64>,
    diffs: Option<Vec<f64>>,
    variance: Option<f64>,
    stdev: Option<f64>
}

impl Dataset {
    fn new(data: Vec<f64>) -> Dataset {
        Dataset {
            data,
            mean: None, 
            diffs: None,
            variance: None,
            stdev: None
        }
    }

    fn mean(&mut self) -> f64 {
        match self.mean {
            Some(mean) => mean, 
            None => {
                let sum: f64 = self.data.iter().sum();
                let mean = sum / self.data.len() as f64;
                self.mean = Some(mean);
                mean
            }
        }
    }

    fn diffs(&mut self) -> &Vec<f64> {
        let result = match self.diffs {
            Some(ref diffs) => &diffs,
            None => {
                let mean = self.mean();
                let diffs = self.data
                    .iter()
                    .map(|x| x - mean)
                    .collect();
                self.diffs = Some(diffs);
                self.diffs()
            }
        };
        &result
    }

    fn variance(&mut self) -> f64 {
        match self.variance {
            Some(variance) => variance, 
            None => {
                let sq_diffs_sum: f64 = self.diffs()
                    .iter()
                    .map(|x| x*x)
                    .sum();
                let variance: f64 = sq_diffs_sum / (self.data.len() as f64 - 1.0);
                self.variance = Some(variance);
                variance
            }
        }
    }

    fn stdev(&mut self) -> f64 {
        match self.stdev {
            Some(stdev) => stdev, 
            None => {
                let stdev: f64 = self.variance().sqrt();
                self.stdev = Some(stdev);
                stdev
            }
        }
    }
}

#[derive(Debug)]
pub struct Linear {
    pub x: Dataset,
    pub y: Dataset,
    pub gradient: f64,
    pub intercept: f64,
    pub covariance: f64
}

impl Linear {
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Linear {
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
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_calculates_mean() {
        let x: Vec<f64> = (1..11).map(|x| x as f64).collect();
        let mut dataset = Dataset::new(x);
        assert_eq!(5.5, dataset.mean());
    }
    
    #[test]
    fn it_calculates_standard_deviation() {
        let x: Vec<f64> = (1u32..11u32).map(|x| x as f64).collect();
        let mut dataset = Dataset::new(x);
        let dps = 100000.0;
        assert_eq!(3.02765, (dataset.stdev() * dps).round() / dps);
    }

    #[test]
    fn it_generates_a_correct_linear_regression_for_simple_data() {
        let x: Vec<f64> = (1u32..7u32).map(|x| x as f64).collect();
        // y = 3x + 4
        let y: Vec<f64> = x.iter().map(|x| 3.0*x + 4.0).collect(); 
        let results = Linear::new(x, y);

        assert_eq!(3.0, results.gradient);
        assert_eq!(4.0, results.intercept);
    }
}
