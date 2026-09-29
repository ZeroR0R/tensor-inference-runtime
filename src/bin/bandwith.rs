// Bandwith Micro-Benchmark
use std::hint::black_box;
use std::time::Instant;

pub fn main() {
    let size = 1_000_000_000;
    let mut run_count = 10;
    let mut runs = vec![];

    // f64 is 8 bytes, 8 read + 8 write
    let traffic = (size * 16) as f64 / 1_000_000_000.0;

    while run_count != 0 {
        runs.push(bandwith(size));
        run_count -= 1
    }

    runs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let median = runs[runs.len() / 2];

    let bandwith = traffic / median;

    println!("Bandwith: {bandwith} GB/s");
    println!(
        "Min: {:?}, Max: {:?}",
        traffic / runs.last().unwrap(),
        traffic / runs.first().unwrap()
    )
}

fn bandwith(size: usize) -> f64 {
    let a = vec![5.0; size];
    // We black box this so the compiler doesn't optimize it away
    let mut b = black_box(vec![0.0; size]);

    let now = Instant::now();

    b.copy_from_slice(&a);

    let total = now.elapsed().as_secs_f64();
    // I didn't see a defensible difference with this black box,
    // but theoretically by making the contents observable after the copy
    // the compiler should never assume it's dead
    black_box(b);

    total
}
