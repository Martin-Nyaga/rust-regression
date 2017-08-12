pub struct RegressionResults {
    gradient: f64,
    intercept: f64
}

impl RegressionResults {
    fn print_equation(&self) {
        println!("y = {}x + {}", self.gradient, self.intercept);
    }
}

 pub fn linear(x: &Vec<f64>, y: &Vec<f64>) -> RegressionResults {
    let r = pearsons_correlation(x, y);    
    let intercept = r * (stdev(x) / stdev(y));
    let gradient = mean(y) - r * mean(x);
    
    RegressionResults {
        gradient,
        intercept,
    }
 }


fn pearsons_correlation(x: &Vec<f64>, y: &Vec<f64>) -> f64 {
   covariance(x, y) / (stdev(x) * stdev(y))
}

fn covariance(x: &Vec<f64>, y: &Vec<f64>) -> f64 {
    let x_mean = mean(x);
    let x_diffs: Vec<f64> = x.iter().map(|x1| x1 - x_mean).collect();

    let y_mean = mean(y);
    let y_diffs: Vec<f64> = y.iter().map(|y1| y1 - y_mean).collect();

    let products: Vec<f64> = x_diffs.iter().zip(y_diffs).map(|(x1, y1)| x1 * y1).collect();
    let sum = products.iter().fold(0.0 as f64, |sum, x| sum + x);

    sum / (x.len() as f64 - 1.0)
}

fn stdev(arr: &Vec<f64>) -> f64 {
    let mean = mean(arr);
    let diffs = arr.iter().fold(0.0, |sum, x| sum + (x - mean).powi(2)) as f64;
    let variance = match arr.len() {
        len => diffs / (arr.len() as f64),
        len if len <= 30 => diffs / (arr.len() as f64 - 1.0),
        _ => 0.0 
    };
    variance.sqrt()
}

fn mean(arr: &Vec<f64>) -> f64 {
    let sum = arr.iter().fold(0_f64, |sum, i| sum + i);
    sum / arr.len() as f64
}


#[cfg(test)]
mod tests {
    #[test]
    fn it_calculates_mean() {
        let x: Vec<f64> = (1..11).map(|x| x as f64).collect();
        assert_eq!(5.5, super::mean(&x));
    }
    #[test]
    fn it_generates_a_correct_linear_regression_for_simple_data() {
        let x: Vec<f64> = (1..7).map(|x| x as f64).collect();
        // y = 3x + 4
        let y: Vec<f64> = x.iter().map(|x| 3.0*x + 4.0).collect(); 
        let results = super::linear(&x, &y);
        results.print_equation();

        assert_eq!(3.0, results.gradient);
        assert_eq!(4.0, results.intercept);
    }
}
