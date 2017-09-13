extern crate regression;
extern crate csv;
extern crate gnuplot;

use gnuplot::*;
use std::error::Error;
use std::collections::HashMap;
use regression::*;

fn main() {
    if let Err(e) = run() {
        println!("{}", e);
    }
}

type Record = HashMap<String, String>;

fn run() -> Result<(), Box<Error>> {
    let (xs, ys) = parse_csv(
        "./sample_datasets/polynomial.csv",
        "Temp",
        "Yield"
    ); 
    let reg = regression::Polynomial::new(ys.clone(), xs.clone(), 2);
    println!("{:#?}", reg);

    let mut fg = Figure::new();
    fg.axes2d()
        .points(
            &xs,
            &ys,
            &[
                Caption("data"),
                PointSymbol('x'),
            ])
        .points(
            &xs,
            &reg.predictions(),
            &[
                Caption("Regression"),
                Color("green"),
                PointSymbol('*')
            ]);
    fg.show();

    Ok(())
}

fn parse_csv(filepath: &str, x_header: &str, y_header: &str) -> (Vec<f64>, Vec<f64>) {
    let mut reader = csv::Reader::from_path(filepath).expect("Failed to open file");

    let mut xs = Vec::new();
    let mut ys = Vec::new();

    for record in reader.deserialize() {
        let record: Record = record.expect("Failed to read record");
        let x = match record.get(x_header).unwrap().parse::<f64>() {
            Ok(num) => num,
            Err(_) => continue
        };
        let y = match record.get(y_header).unwrap().parse::<f64>() {
            Ok(num) => num,
            Err(_) => continue
        };
        xs.push(x);
        ys.push(y);
    }

    (xs, ys)
}
