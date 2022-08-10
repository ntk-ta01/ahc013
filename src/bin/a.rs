use rand::prelude::*;

const TIMELIMIT: f64 = 2.8;
const DIJ: [(usize, usize); 4] = [(0, !0), (!0, 0), (0, 1), (1, 0)];
// const DIR: [char; 4] = ['L', 'U', 'R', 'D'];

fn main() {
    let timer = Timer::new();
    let input = Input::new();
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);

    let mut grid = input.grid.clone();
    let mut output_move: Vec<(usize, usize, usize, usize)> = vec![];

    let mut output_connect = vec![];
    let mut uf = UnionFind::new(input.n * input.n);
    let mut cabled_grid = grid.clone();
    for i in 0..input.n {
        for j in 0..input.n {
            if cabled_grid[i][j] == 0 || cabled_grid[i][j] == !0 {
                continue;
            }
            for &(di, dj) in DIJ.iter() {
                let mut ni = i;
                let mut nj = j;
                for len in 1..input.n {
                    ni += di;
                    nj += dj;
                    if input.n <= ni || input.n <= nj {
                        break;
                    }
                    if cabled_grid[ni][nj] == !0 {
                        break;
                    }
                    if cabled_grid[ni][nj] == 0 {
                        continue;
                    }
                    if uf.same(i * input.n + j, ni * input.n + nj) {
                        break;
                    }
                    if cabled_grid[ni][nj] == cabled_grid[i][j] {
                        output_connect.push((i, j, ni, nj));
                        uf.unite(i * input.n + j, ni * input.n + nj);
                        for _ in 0..len - 1 {
                            ni -= di;
                            nj -= dj;
                            cabled_grid[ni][nj] = !0; // cableを引いたマス
                        }
                    }
                    break;
                }
            }
        }
    }

    let mut now_score = 0;
    let mut pos = vec![];
    for (r, g_row) in grid.iter().enumerate() {
        for (c, &g) in g_row.iter().enumerate() {
            if g != !0 && g != 0 {
                pos.push((r, c));
            }
        }
    }

    for (i, &(ri, ci)) in pos.iter().enumerate() {
        for &(rj, cj) in pos.iter().skip(i + 1) {
            if uf.same(ri * input.n + ci, rj * input.n + cj) {
                if grid[ri][ci] == grid[rj][cj] {
                    now_score += 1;
                } else {
                    now_score -= 1;
                }
            }
        }
    }
    println!("{}", output_move.len());
    for (a, b, c, d) in output_move.iter() {
        println!("{} {} {} {}", a, b, c, d);
    }
    println!("{}", output_connect.len());
    for (e, f, g, h) in output_connect.iter() {
        println!("{} {} {} {}", e, f, g, h);
    }
    // eprintln!("{}", now_score);
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
    fn new(n: usize) -> Self {
        UnionFind {
            par: (0..n).into_iter().collect(),
            size: vec![1; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.par[x] == x {
            x
        } else {
            self.par[x] = self.find(self.par[x]);
            self.par[x]
        }
    }

    fn unite(&mut self, x: usize, y: usize) {
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

    fn same(&mut self, x: usize, y: usize) -> bool {
        self.find(x) == self.find(y)
    }
}

fn get_time() -> f64 {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    t.as_secs() as f64 + t.subsec_nanos() as f64 * 1e-9
}

struct Timer {
    start_time: f64,
}

impl Timer {
    fn new() -> Timer {
        Timer {
            start_time: get_time(),
        }
    }

    fn get_time(&self) -> f64 {
        get_time() - self.start_time
    }
}
