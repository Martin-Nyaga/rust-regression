extern crate regression;
extern crate csv;
extern crate gnuplot;

use gnuplot::*;
use regression::Linear;
use std::error::Error;
use std::collections::HashMap;

fn main() {
    if let Err(e) = run() {
        println!("{}", e);
    }
}

type Record = HashMap<String, String>;

fn run() -> Result<(), Box<Error>> {
    let (xs, ys) = parse_csv("./sample_datasets/countries.csv"); 
    let reg = Linear::new(&xs, &ys);

    let mut fg = Figure::new();
    fg.axes2d()
        .points(
            &xs,
            &ys,
            &[
                Caption("Life Expectancy vs Infant Mortality"),
                PointSymbol('x'),
            ])
        .lines(
            &xs,
            &reg.predictions(),
            &[
                Caption("Trend"),
                Color("green")
            ]);
    fg.show();

    Ok(())
}

fn parse_csv(filepath: &str) -> (Vec<f64>, Vec<f64>) {
    let mut reader = csv::Reader::from_path(filepath).expect("Failed to open file");

    let mut xs = Vec::new();
    let mut ys = Vec::new();

    for record in reader.deserialize() {
        let record: Record = record.expect("Failed to read record");

        let x = match record.get("Infant mortality rate(deaths/1000 live births)").unwrap().parse::<f64>() {
            Ok(num) => num,
            Err(_) => continue
        };
        let y = match record.get("Life expectancy at birth(years)").unwrap().parse::<f64>() {
            Ok(num) => num,
            Err(_) => continue
        };

        xs.push(x);
        ys.push(y);
    }

    (xs, ys)
}
