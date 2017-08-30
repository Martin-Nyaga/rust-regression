pub mod linear;
pub use self::linear::*;

pub mod exponential;
pub use self::exponential::*;

pub mod power;
pub use self::power::*;

pub mod multiple_linear;
pub use self::multiple_linear::*;

// Methods available on all predictions
pub trait Regression {
    fn equation(&self) -> String;

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

pub trait MultipleRegression {
    fn x_data(&self) -> Vec<&Vec<f64>>;
    fn predict_single(&self, x: &Vec<f64>) -> f64;

    // Predicted y values from x data
    fn predictions(&self) -> Vec<f64> {
        let mut predictions = Vec::new();
        for (i, x) in self.x_data()[0].iter().enumerate() {
            let arr = self.x_data().iter().map(|x_arr| x_arr[i]).collect(); 
            predictions.push(self.predict_single(&arr));
        }
        predictions
    }

    // // Predicted y values from any provided x data
    // fn predict_multi(&self, x: &Vec<f64>) -> Vec<f64> {
    //     x.iter()
    //         .map(|x| self.predict_single(x))
    //         .collect()
    // }

    // Return r2 of the regression
    fn r2(&self) -> f64 {
        // TODO
        0.0
    }
}
