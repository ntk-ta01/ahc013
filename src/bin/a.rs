use rand::prelude::*;

const TIMELIMIT: f64 = 2.75;
const DIJ: [(usize, usize); 4] = [(0, !0), (!0, 0), (0, 1), (1, 0)];
// const DIR: [char; 4] = ['L', 'U', 'R', 'D'];
type Output = Vec<(usize, usize, usize, usize)>;

fn main() {
    let timer = Timer::new();
    let input = Input::new();
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
    let (mut grid, mut computers) = {
        let mut g = vec![vec![Cell::Empty; input.n]; input.n];
        let mut cs = vec![];
        for (i, (in_row, row)) in input.grid.iter().zip(g.iter_mut()).enumerate() {
            for (j, (&in_element, element)) in in_row.iter().zip(row.iter_mut()).enumerate() {
                if in_element != 0 {
                    *element = Cell::Computer { index: cs.len() };
                    cs.push(Computer {
                        posi: (i, j),
                        kind: in_element,
                    });
                }
            }
        }
        (g, cs)
    };
    let (output_move, mut output_connect) =
        annealing(&input, &timer, &mut rng, &mut grid, &mut computers);

    println!("{}", output_move.len());
    for (a, b, c, d) in output_move.iter() {
        println!("{} {} {} {}", a, b, c, d);
    }
    while 100 * input.k < output_move.len() + output_connect.len() {
        output_connect.pop();
    }
    println!("{}", output_connect.len());
    for (e, f, g, h) in output_connect.iter() {
        println!("{} {} {} {}", e, f, g, h);
    }
}

fn annealing(
    input: &Input,
    timer: &Timer,
    rng: &mut rand_chacha::ChaCha20Rng,
    start_grid: &mut [Vec<Cell>],
    start_computers: &mut [Computer],
) -> (Output, Output) {
    const T0: f64 = 50.0;
    const T1: f64 = 0.01;
    let mut temp = T0;
    let mut prob;

    let mut moves = vec![];
    let mut connects = vec![];
    let (mut score, mut best_connects) = {
        let mut grid = start_grid.to_owned();
        let mut computers = start_computers.to_owned();
        for _ in 0..input.k * 15 {
            let mut moveable = vec![];
            for (i, computer) in computers.iter().enumerate() {
                for &(di, dj) in DIJ.iter() {
                    let ni = computer.posi.0 + di;
                    let nj = computer.posi.1 + dj;
                    if input.n <= ni || input.n <= nj {
                        continue;
                    }
                    match grid[ni][nj] {
                        Cell::Empty => moveable.push((i, (ni, nj))),
                        Cell::Cable { kind } => {
                            if kind == computer.kind {
                                moveable.push((i, (ni, nj)));
                            }
                        }
                        Cell::Computer { index: _ } => {}
                    }
                }
            }
            if moveable.is_empty() {
                break;
            }
            let (com_i, next) = moveable[rng.gen_range(0, moveable.len())];
            if !computers[com_i].go(input, next, &mut grid) {
                unreachable!();
            }
            moves.push((com_i, next));
        }

        compute_score(input, &mut grid, &mut computers, moves.len())
    };

    let mut best_score = score;
    let mut best_moves = moves.clone();
    // movesの順にcomputerを動かす
    // どの向きに動かすかは探索する（今はランダム）
    let mut count = 0;
    // let mut neigh0 = 0;
    // let mut neigh1 = 0;
    'lp: loop {
        if count >= 100 {
            let passed = timer.get_time() / TIMELIMIT;
            if passed >= 1.0 {
                break;
            }
            temp = T0.powf(1.0 - passed) * T1.powf(passed);
            count = 0;
        }
        count += 1;
        let mut new_moves = moves.clone();
        let mut new_grid = start_grid.to_owned();
        let mut new_computers = start_computers.to_owned();
        // 近傍解作成
        let neigh_count = 2;
        let neigh = rng.gen_range(0, neigh_count);
        match neigh {
            0 => {
                // insert
                if 40 * input.k <= new_moves.len() {
                    continue 'lp;
                }
                for _ in 0..rng.gen_range(1, 5) {
                    if 40 * input.k < new_moves.len() {
                        break;
                    }
                    let mut new = vec![];
                    new_grid = start_grid.to_owned();
                    new_computers = start_computers.to_owned();
                    let insert_i = if new_moves.is_empty() {
                        0
                    } else {
                        rng.gen_range(0, new_moves.len())
                    };
                    for &(com_i, next) in new_moves.iter().take(insert_i) {
                        if !new_computers[com_i].go(input, next, &mut new_grid) {
                            unreachable!()
                        }
                        new.push((com_i, next));
                    }
                    let mut moveable = vec![];
                    for (i, computer) in new_computers.iter().enumerate() {
                        for &(di, dj) in DIJ.iter() {
                            let ni = computer.posi.0 + di;
                            let nj = computer.posi.1 + dj;
                            if input.n <= ni || input.n <= nj {
                                continue;
                            }
                            match new_grid[ni][nj] {
                                Cell::Empty => moveable.push((i, (ni, nj))),
                                Cell::Computer { index: _ } => {}
                                Cell::Cable { kind: _ } => {
                                    unreachable!();
                                }
                            }
                        }
                    }
                    if moveable.is_empty() {
                        continue;
                    }
                    let (com_i, next) = moveable[rng.gen_range(0, moveable.len())];
                    new_computers[com_i].go(input, next, &mut new_grid);
                    new_moves.insert(insert_i, (com_i, next));
                    new.push((com_i, next));
                    for &(com_i, next) in new_moves.iter().skip(insert_i) {
                        if new_computers[com_i].go(input, next, &mut new_grid) {
                            new.push((com_i, next));
                        }
                    }
                    new_moves = new;
                }
            }
            1 => {
                // remove
                for _ in 0..rng.gen_range(1, 10) {
                    if new_moves.is_empty() {
                        break;
                    }
                    let i = rng.gen_range(0, new_moves.len());
                    new_moves.remove(i);
                }
                for &(com_i, next) in new_moves.iter() {
                    if !new_computers[com_i].go(input, next, &mut new_grid) {
                        continue 'lp;
                    }
                }
            }
            _ => unreachable!(),
        }

        // 近傍解作成ここまで
        let (new_score, new_connect) =
            compute_score(input, &mut new_grid, &mut new_computers, new_moves.len());
        prob = f64::exp((new_score - score) as f64 / temp);
        if score < new_score || rng.gen_bool(prob) {
            score = new_score;
            moves = new_moves;
            connects = new_connect;
        }

        if best_score < score {
            // if neigh == 0 {
            //     neigh0 += 1;
            // }
            // if neigh == 1 {
            //     neigh1 += 1;
            // }
            best_score = score;
            best_moves = moves.clone();
            best_connects = connects.clone();
        }
    }
    // eprintln!("insert: {} remove: {}", neigh0, neigh1);
    // eprintln!("count: {}", count);
    eprintln!("best_score: {}", best_score);
    let moves = {
        let mut mv = vec![];
        let grid = start_grid;
        let computers = start_computers;
        for (com_i, next) in best_moves {
            let pos = computers[com_i].posi;
            mv.push((pos.0, pos.1, next.0, next.1));
            computers[com_i].go(input, next, grid);
        }
        mv
    };
    (moves, best_connects)
}

fn compute_score(
    input: &Input,
    grid: &mut [Vec<Cell>],
    computers: &mut [Computer],
    move_time: usize,
) -> (i64, Vec<(usize, usize, usize, usize)>) {
    let mut score = 0;
    let mut output_connect = vec![];
    let mut uf = UnionFind::new(input.n * input.n);
    'connect_lp: for i in 0..input.n {
        for j in 0..input.n {
            if grid[i][j].is_cable() || grid[i][j].is_empty() {
                continue;
            }
            let now_computer = if let Cell::Computer { index } = grid[i][j] {
                &computers[index]
            } else {
                unreachable!()
            };
            for &(di, dj) in DIJ.iter() {
                let mut ni = i;
                let mut nj = j;
                for len in 1..input.n {
                    ni += di;
                    nj += dj;
                    if input.n <= ni || input.n <= nj {
                        break;
                    }
                    if let Cell::Cable { kind: _ } = grid[ni][nj] {
                        break;
                    }
                    if Cell::Empty == grid[ni][nj] {
                        continue;
                    }
                    if uf.same(i * input.n + j, ni * input.n + nj) {
                        break;
                    }
                    let next_computer = if let Cell::Computer { index } = grid[ni][nj] {
                        &computers[index]
                    } else {
                        unreachable!()
                    };
                    if now_computer.kind == next_computer.kind {
                        output_connect.push((i, j, ni, nj));
                        uf.unite(i * input.n + j, ni * input.n + nj);
                        for _ in 0..len - 1 {
                            ni -= di;
                            nj -= dj;
                            grid[ni][nj] = Cell::Cable {
                                kind: now_computer.kind,
                            };
                        }
                    }
                    if 100 * input.k == move_time + output_connect.len() {
                        break 'connect_lp;
                    }
                    break;
                }
            }
        }
    }

    let mut pos = vec![];
    for (r, g_row) in grid.iter().enumerate() {
        for (c, &g) in g_row.iter().enumerate() {
            if g.is_computer() {
                pos.push((r, c));
            }
        }
    }

    for (i, &(ri, ci)) in pos.iter().enumerate() {
        let computer1 = if let Cell::Computer { index } = grid[ri][ci] {
            &computers[index]
        } else {
            unreachable!()
        };
        for &(rj, cj) in pos.iter().skip(i + 1) {
            let computer2 = if let Cell::Computer { index } = grid[rj][cj] {
                &computers[index]
            } else {
                unreachable!()
            };
            if uf.same(ri * input.n + ci, rj * input.n + cj) {
                if computer1.kind == computer2.kind {
                    score += 1;
                } else {
                    score -= 1;
                }
            }
        }
    }
    (score, output_connect)
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Cell {
    Empty,
    Computer { index: usize },
    Cable { kind: usize },
}

impl Cell {
    fn is_empty(&self) -> bool {
        matches!(*self, Cell::Empty)
    }

    fn is_computer(&self) -> bool {
        matches!(*self, Cell::Computer { index: _ })
    }

    fn is_cable(&self) -> bool {
        matches!(*self, Cell::Cable { kind: _ })
    }
}

#[derive(Debug, Clone, Copy)]
struct Computer {
    posi: (usize, usize),
    kind: usize,
}

impl Computer {
    fn go(&mut self, input: &Input, next: (usize, usize), grid: &mut [Vec<Cell>]) -> bool {
        if grid[next.0][next.1] != Cell::Empty {
            return false;
        }
        if DIJ.iter().all(|&(di, dj)| {
            self.posi.0 + di >= input.n
                || self.posi.1 + dj >= input.n
                || (self.posi.0 + di, self.posi.1 + dj) != next
        }) {
            return false;
        }
        grid[next.0][next.1] = grid[self.posi.0][self.posi.1];
        grid[self.posi.0][self.posi.1] = Cell::Empty;
        self.posi = next;
        true
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compute_score_works() {
        let input = Input {
            n: 5,
            k: 3,
            grid: vec![
                vec![0, 2, 0, 2, 0],
                vec![2, 1, 0, 2, 3],
                vec![3, 0, 3, 0, 2],
                vec![0, 1, 0, 3, 3],
                vec![3, 2, 0, 3, 2],
            ],
        };
        let (mut grid, mut computers) = {
            let mut g = vec![vec![Cell::Empty; input.n]; input.n];
            let mut cs = vec![];
            for (i, (in_row, row)) in input.grid.iter().zip(g.iter_mut()).enumerate() {
                for (j, (&in_element, element)) in in_row.iter().zip(row.iter_mut()).enumerate() {
                    if in_element != 0 {
                        *element = Cell::Computer { index: cs.len() };
                        cs.push(Computer {
                            posi: (i, j),
                            kind: in_element,
                        });
                    }
                }
            }
            (g, cs)
        };
        let (score, output_connect) = compute_score(&input, &mut grid, &mut computers, 0);
        println!("{}", output_connect.len());
        for (e, f, g, h) in output_connect.iter() {
            println!("{} {} {} {}", e, f, g, h);
        }
        println!("score: {}", score);
    }
}
