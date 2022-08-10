use rand::prelude::*;

const TIMELIMIT: f64 = 2.8;
const DIJ: [(usize, usize); 4] = [(0, !0), (!0, 0), (0, 1), (1, 0)];
// const DIR: [char; 4] = ['L', 'U', 'R', 'D'];

fn main() {
    let timer = Timer::new();
    let input = Input::new();
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);

    let mut max_move = vec![];
    let mut max_connect = vec![];
    let mut max_score = 0;
    let mut grid = input.grid.clone();
    let mut output_move = vec![
        (8, 3, 9, 3),
        (1, 8, 1, 9),
        (9, 13, 9, 14),
        (10, 13, 9, 13),
        (10, 11, 10, 12),
        (10, 12, 10, 13),
        (11, 12, 10, 12),
        (12, 12, 11, 12),
        (12, 11, 12, 12),
        (10, 10, 10, 11),
        (10, 9, 10, 10),
        (4, 6, 4, 7),
        (4, 5, 4, 6),
        (4, 4, 4, 5),
        (4, 3, 4, 4),
        (0, 11, 1, 11),
        (0, 10, 0, 11),
        (1, 10, 0, 10),
        (1, 9, 1, 10),
        (1, 7, 1, 8),
        (1, 8, 1, 9),
        (0, 6, 1, 6),
        (0, 7, 1, 7),
        (0, 8, 1, 8),
    ];
    grid[9][3] = grid[8][3];
    grid[8][3] = 0;
    grid[1][9] = grid[1][8];
    grid[1][8] = 0;
    grid[9][14] = grid[9][13];
    grid[9][13] = 0;
    grid[9][13] = grid[10][13];
    grid[10][13] = 0;
    grid[10][12] = grid[10][11];
    grid[10][11] = 0;
    grid[10][13] = grid[10][12];
    grid[10][12] = 0;
    grid[10][12] = grid[11][12];
    grid[11][12] = 0;
    grid[11][12] = grid[12][12];
    grid[12][12] = 0;
    grid[12][12] = grid[12][11];
    grid[12][11] = 0;
    grid[10][11] = grid[10][10];
    grid[10][10] = 0;
    grid[10][10] = grid[10][9];
    grid[10][9] = 0;
    grid[4][7] = grid[4][6];
    grid[4][6] = 0;
    grid[4][6] = grid[4][5];
    grid[4][5] = 0;
    grid[4][5] = grid[4][4];
    grid[4][4] = 0;
    grid[4][4] = grid[4][3];
    grid[4][3] = 0;
    grid[1][11] = grid[0][11];
    grid[0][11] = 0;
    grid[0][11] = grid[0][10];
    grid[0][10] = 0;
    grid[0][10] = grid[1][10];
    grid[1][10] = 0;
    grid[1][10] = grid[1][9];
    grid[1][9] = 0;
    grid[1][8] = grid[1][7];
    grid[1][7] = 0;
    grid[1][9] = grid[1][8];
    grid[1][8] = 0;
    grid[1][6] = grid[0][6];
    grid[0][6] = 0;
    grid[1][7] = grid[0][7];
    grid[0][7] = 0;
    grid[1][8] = grid[0][8];
    grid[0][8] = 0;
    for _ in 0..100 * input.k {
        let mut nowi = 0;
        let mut nowj = 0;
        'lp: for i in 0..input.n {
            for j in 0..input.n {
                if grid[i][j] != 0
                    && DIJ.iter().any(|&(di, dj)| {
                        i + di < input.n && j + dj < input.n && grid[i + di][j + dj] == 0
                    })
                {
                    nowi = i;
                    nowj = j;
                    break 'lp;
                }
            }
        }
        let mut dir = rng.gen_range(0, 4);
        while nowi + DIJ[dir].0 >= input.n
            || nowj + DIJ[dir].1 >= input.n
            || grid[nowi + DIJ[dir].0][nowj + DIJ[dir].1] != 0
        {
            dir += 1;
            dir %= 4;
        }
        output_move.push((nowi, nowj, nowi + DIJ[dir].0, nowj + DIJ[dir].1));
        grid[nowi + DIJ[dir].0][nowj + DIJ[dir].1] = grid[nowi][nowj];
        grid[nowi][nowj] = 0;

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
        if 100 * input.k < output_move.len() + output_connect.len() {
            continue;
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
        if max_score < now_score {
            max_score = now_score;
            max_move = output_move.clone();
            max_connect = output_connect.clone();
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
