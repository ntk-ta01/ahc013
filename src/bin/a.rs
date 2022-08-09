use rand::prelude::*;

const TIMELIMIT: f64 = 2.8;
const DIJ: [(usize, usize); 4] = [(0, 1), (0, !0), (1, 0), (!0, 0)];

fn main() {
    let timer = Timer::new();
    let input = Input::new();

    let mut max_move = vec![];
    let mut max_connect = vec![];
    let mut max_score = 0;
    for seed in 0.. {
        if seed % 10 == 0 && timer.get_time() > TIMELIMIT {
            break;
        }
        let mut grid = input.grid.clone();
        let mut output_move = vec![];
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(seed);
        for _ in 0..100 * input.k {
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

            let mut output_connect = vec![];
            let mut uf = UnionFind::new(input.n * input.n);
            for i in 0..input.n {
                for j in 0..input.n {
                    if grid[i][j] == 0 || grid[i][j] == !0 {
                        continue;
                    }
                    for &(di, dj) in DIJ.iter() {
                        let mut ni = i;
                        let mut nj = j;
                        for len in 1..2 {
                            ni += di;
                            nj += dj;
                            if input.n <= ni || input.n <= nj {
                                break;
                            }
                            if grid[ni][nj] == 0 || grid[ni][nj] == !0 {
                                continue;
                            }
                            if uf.same(i * input.n + j, ni * input.n + nj) {
                                break;
                            }
                            if grid[ni][nj] == grid[i][j] {
                                output_connect.push((i, j, ni, nj));
                                uf.unite(i * input.n + j, ni * input.n + nj);
                                for _ in 0..len - 1 {
                                    ni -= di;
                                    nj -= dj;
                                    grid[ni][nj] = input.n; // cableを引いたマス
                                }
                            }
                            break;
                        }
                    }
                }
            }
            if 100 * input.k < output_move.len() + output_connect.len() {
                break;
            }

            let mut now_score = 0;
            let mut pos = vec![];
            for r in 0..input.n {
                for c in 0..input.n {
                    if grid[r][c] != !0 && grid[r][c] != 0 {
                        pos.push((r, c));
                    }
                }
            }

            let computers = pos.len();
            for i in 0..computers {
                let (ri, ci) = pos[i];
                for j in i + 1..computers {
                    let (rj, cj) = pos[j];
                    if uf.same(ri * input.n + ci, rj * input.n + cj) {
                        if grid[ri][ci] == grid[rj][cj] {
                            now_score += 1;
                        } else {
                            now_score -= 1;
                        }
                    }
                }
            }
            if max_score < now_score {
                max_score = now_score;
                max_move = output_move.clone();
                max_connect = output_connect.clone();
            }
        }
    }
    println!("{}", max_move.len());
    for (a, b, c, d) in max_move.iter() {
        println!("{} {} {} {}", a, b, c, d);
    }
    println!("{}", max_connect.len());
    for (e, f, g, h) in max_connect.iter() {
        println!("{} {} {} {}", e, f, g, h);
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
