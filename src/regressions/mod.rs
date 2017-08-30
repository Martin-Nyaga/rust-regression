pub mod linear;
pub use self::linear::*;

pub mod exponential;
pub use self::exponential::*;

// Methods available on all predictions
pub trait Regression {
    fn x_data(&self) -> &Vec<f64>;
    fn predict_single(&self, x: f64) -> f64;

    // Predicted y values from x data
    fn predictions(&self) -> Vec<f64> {
        self.x_data()
            .iter()
            .map(|x| self.predict_single(x.to_owned()))
            .collect()
    }

    // Predicted y values from any provided x data
    fn predict_multi(&self, x: &Vec<f64>) -> Vec<f64> {
        x.iter()
            .map(|x| self.predict_single(x.to_owned()))
            .collect()
    }

    // Return r2 of the regression
    fn r2(&self) -> f64 {
        // TODO
        0.0
    }
}
