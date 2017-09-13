use utils::Dataset;

pub mod linear;
pub use self::linear::*;

pub mod exponential;
pub use self::exponential::*;

pub mod power;
pub use self::power::*;

pub mod polynomial;
pub use self::polynomial::*;

pub mod multiple_linear;
pub use self::multiple_linear::*;

pub trait Regression {
    fn x_data(&self) -> &Dataset;
    fn y_data(&self) -> &Dataset;
    fn equation_string(&self) -> String;
    fn predict_single(&self, x: f64) -> f64;

    fn predictions(&self) -> Vec<f64> {
        self.x_data()
            .data.iter()
            .map(|x| self.predict_single(x.to_owned()))
            .collect()
    }

    fn residuals(&self) -> Vec<f64> {
        self.y_data()
            .differences(&self.predictions())
    }

    fn mean_square_error(&self) -> f64 {
        self.residuals()
            .iter()
            .map(|x| x * x)
            .sum()
    }

    fn predict_multi(&self, x: &Vec<f64>) -> Vec<f64> {
        x.iter()
            .map(|x| self.predict_single(x.to_owned()))
            .collect()
    }
}

