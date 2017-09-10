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
    // let (xs, ys) = parse_csv("./sample_datasets/countries.csv"); 
    
    let xs: Vec<f64> = (1u32..30u32).map(|x| x as f64).collect();
    let x2s: Vec<f64> = xs.iter().map(|x| x*x).collect();
    let ys: Vec<f64> = xs.iter()
        .map(|x| x.powi(2) + x)
        .collect(); 

    let x_arr = vec![&x2s, &xs];
    let reg = regression::MultipleLinear::new(&ys, &x_arr);
    println!("{:?}", reg);

    let mut fg = Figure::new();
    fg.axes2d()
        .points(
            &xs,
            &ys,
            &[
                Caption("x^2 + x"),
                PointSymbol('x'),
            ])
        .points(
            &xs,
            &reg.predictions(),
            &[
                Caption(&gnuplot_format_string(reg.equation_string())),
                Color("green"),
                PointSymbol('*')
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

fn gnuplot_format_string(s: String) -> String {
    s.replace("(", "{").replace(")","}") 
}
