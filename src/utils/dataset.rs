#[derive(Debug)]
pub struct Dataset<'a> {
    pub data: &'a Vec<f64>,
    mean: Option<f64>,
    diffs: Option<Vec<f64>>,
    variance: Option<f64>,
    stdev: Option<f64>
}

impl<'a> Dataset<'a> {
    pub fn new(data: &'a Vec<f64>) -> Dataset<'a> {
        Dataset {
            data,
            mean: None, 
            diffs: None,
            variance: None,
            stdev: None
        }
    }

    pub fn mean(&mut self) -> f64 {
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

    pub fn diffs(&mut self) -> &Vec<f64> {
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

    pub fn variance(&mut self) -> f64 {
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

    pub fn stdev(&mut self) -> f64 {
        match self.stdev {
            Some(stdev) => stdev, 
            None => {
                let stdev: f64 = self.variance().sqrt();
                self.stdev = Some(stdev);
                stdev
            }
        }
    }

    pub fn residuals(&self, predictions: Vec<f64>) -> Vec<f64> {
        self.data.iter()
            .zip(predictions)
            .map(|(a,b)| a - b)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_calculates_mean() {
        let x: Vec<f64> = (1..11).map(|x| x as f64).collect();
        let mut dataset = Dataset::new(&x);
        assert_eq!(5.5, dataset.mean());
    }
    
    #[test]
    fn it_calculates_standard_deviation() {
        let x: Vec<f64> = (1u32..11u32).map(|x| x as f64).collect();
        let mut dataset = Dataset::new(&x);
        let dps = 100000.0;
        assert_eq!(3.02765, (dataset.stdev() * dps).round() / dps);
    }
}
