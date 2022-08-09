use rand::prelude::*;

const TIMELIMIT: f64 = 2.8;
const DIJ: [(usize, usize); 4] = [(0, 1), (0, !0), (1, 0), (!0, 0)];

fn main() {
    let input = Input::new();
    let mut grid = input.grid.clone();

    let mut output_move = vec![];
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
    for turn in 0..100 * input.k {
        let mut i = rng.gen_range(0, input.n);
        let mut j = rng.gen_range(0, input.n);
        while grid[i][j] == 0
            || DIJ.iter().all(|&(di, dj)| {
                i + di >= input.n || j + dj >= input.n || grid[i + di][j + dj] != 0
            })
        {
            i = rng.gen_range(0, input.n);
            j = rng.gen_range(0, input.n);
        }
        let mut dir = rng.gen_range(0, 4);
        while i + DIJ[dir].0 >= input.n
            || j + DIJ[dir].1 >= input.n
            || grid[i + DIJ[dir].0][j + DIJ[dir].1] != 0
        {
            dir += 1;
            dir %= 4;
        }
        output_move.push((i, j, i + DIJ[dir].0, j + DIJ[dir].1));
        grid[i + DIJ[dir].0][j + DIJ[dir].1] = grid[i][j];
        grid[i][j] = 0;
        println!("{}", output_move.len());
        for (a, b, c, d) in output_move.iter() {
            println!("{} {} {} {}", a, b, c, d);
        }
        println!("0");
    }
}

#[derive(Debug)]
struct Input {
    n: usize,
    k: usize,
    grid: Vec<Vec<usize>>,
}

impl Input {
    fn new() -> Self {
        use proconio::{input, marker::Chars};
        input! {
            n: usize,
            k: usize,
            grid: [Chars; n]
        }
        let grid = grid
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&c| c.to_string().parse::<usize>().unwrap())
                    .collect()
            })
            .collect();
        Input { n, k, grid }
    }
}

#[derive(Debug)]
pub struct UnionFind {
    par: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        UnionFind {
            par: (0..n).into_iter().collect(),
            size: vec![1; n],
        }
    }

    pub fn find(&mut self, x: usize) -> usize {
        if self.par[x] == x {
            x
        } else {
            self.par[x] = self.find(self.par[x]);
            self.par[x]
        }
    }

    pub fn unite(&mut self, x: usize, y: usize) {
        let mut x = self.find(x);
        let mut y = self.find(y);
        if self.size[x] < self.size[y] {
            std::mem::swap(&mut x, &mut y);
        }
        if x != y {
            self.size[x] += self.size[y];
            self.par[y] = x;
        }
    }

    pub fn same(&mut self, x: usize, y: usize) -> bool {
        self.find(x) == self.find(y)
    }
}
