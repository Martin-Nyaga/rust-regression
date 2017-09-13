use utils::Dataset;
use regressions::{Regression, MultipleLinear};

#[derive(Debug)]
pub struct Polynomial {
    x: Dataset,
    y: Dataset,
    order: u8,
    pub coefficients: Vec<f64>,
    pub intercept: f64
}

impl Polynomial {
    pub fn new(y: Vec<f64>, x: Vec<f64>, order: u8) -> Polynomial {
        let x_arr: Vec<Vec<f64>> =
            (0i32..order as i32).map(|i| {
                x.clone()
                    .iter()
                    .map(|x1| x1.to_owned().powi(i + 1))
                    .collect::<Vec<f64>>()
            }).collect();

        println!("{:?}", x_arr);

        let multiple_reg = MultipleLinear::new(y.clone(), x_arr);
        println!("{:#?}", multiple_reg);

        Polynomial {
            x: Dataset::new(x),
            y: Dataset::new(y),
            order: order,
            coefficients: multiple_reg.coefficients,
            intercept: multiple_reg.intercept
        }
    }
}

impl Regression for Polynomial {
    fn x_data(&self) -> &Dataset {
        &self.x
    }

    fn y_data(&self) -> &Dataset {
        &self.y
    }
    
    fn predict_single(&self, x: f64) -> f64 {
        self.intercept + 
            self.coefficients
                .iter()
                .enumerate()
                .map(|(i, coeff)| coeff * x.powi((i + 1) as i32))
                .sum::<f64>()
    }

    fn equation_string(&self) -> String {
        format!("{} {:?}", self.intercept, self.coefficients)
    }
}
