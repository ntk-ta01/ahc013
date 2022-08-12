use rand::prelude::*;

const TIMELIMIT: f64 = 2.0;
const DIJ: [(usize, usize); 4] = [(0, !0), (!0, 0), (0, 1), (1, 0)];
// const DIR: [char; 4] = ['L', 'U', 'R', 'D'];

fn main() {
    let timer = Timer::new();
    let input = Input::new();
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
    // greedy(&input);
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
        local_search(&input, &timer, &mut rng, &mut grid, &mut computers);

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

fn greedy(input: &Input) -> i64 {
    let grid = input.grid.clone();
    let output_move: Vec<(usize, usize, usize, usize)> = vec![];

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
    eprintln!("{}", now_score);
    now_score
}

fn local_search(
    input: &Input,
    timer: &Timer,
    rng: &mut rand_chacha::ChaCha20Rng,
    start_grid: &mut Vec<Vec<Cell>>,
    start_computers: &mut Vec<Computer>,
) -> (
    Vec<(usize, usize, usize, usize)>,
    Vec<(usize, usize, usize, usize)>,
) {
    const T0: f64 = 100.0;
    const T1: f64 = 0.01;
    let mut temp = T0;
    let mut prob;

    let mut move_computers = vec![];
    let mut connect_computers = vec![];
    let (mut score, mut best_connect) = {
        let mut grid = start_grid.clone();
        let mut computers = start_computers.clone();
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
            move_computers.push((com_i, next));
        }

        compute_score(input, &mut grid, &mut computers, move_computers.len())
    };

    let mut best_score = score;
    let mut best_move = move_computers.clone();
    // move_computersの順にcomputerを動かす
    // どの向きに動かすかは探索する（今はランダム）
    let mut count = 0;
    'lp: loop {
        if count >= 10 {
            let passed = timer.get_time() / TIMELIMIT;
            if passed >= 1.0 {
                break;
            }
            temp = T0.powf(1.0 - passed) * T1.powf(passed);
            count = 0;
        }
        count += 1;
        let mut new_move_computers = move_computers.clone();
        let mut new_grid = start_grid.clone();
        let mut new_computers = start_computers.clone();
        // 近傍解作成
        let neigh_count = 2;
        let neigh = rng.gen_range(0, neigh_count);
        match neigh {
            0 => {
                // insert
                if 20 * input.k <= new_move_computers.len() {
                    continue 'lp;
                }
                for _ in 0..rng.gen_range(1, 5) {
                    while 20 * input.k < new_move_computers.len() {
                        new_move_computers.pop();
                    }
                    new_grid = start_grid.clone();
                    new_computers = start_computers.clone();
                    let insert_i = if new_move_computers.is_empty() {
                        0
                    } else {
                        rng.gen_range(0, new_move_computers.len())
                    };
                    for &(com_i, next) in new_move_computers.iter().take(insert_i) {
                        if !new_computers[com_i].go(input, next, &mut new_grid) {
                            unreachable!()
                        }
                    }
                    // compute_score(input, &mut new_grid, &mut new_computers);
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
                        continue;
                    }
                    let (com_i, next) = moveable[rng.gen_range(0, moveable.len())];
                    new_move_computers.insert(insert_i, (com_i, next));
                    for &(com_i, next) in new_move_computers.iter().skip(insert_i) {
                        if !new_computers[com_i].go(input, next, &mut new_grid) {
                            continue 'lp;
                        }
                    }
                }
            }
            1 => {
                // remove
                for _ in 0..rng.gen_range(1, 5) {
                    if new_move_computers.is_empty() {
                        continue;
                    }
                    let i = rng.gen_range(0, new_move_computers.len());
                    new_move_computers.remove(i);
                }
                for &(com_i, next) in new_move_computers.iter() {
                    if !new_computers[com_i].go(input, next, &mut new_grid) {
                        continue 'lp;
                    }
                }
            }
            _ => unreachable!(),
        }

        // 近傍解作成ここまで
        let (new_score, new_connect) = compute_score(
            input,
            &mut new_grid,
            &mut new_computers,
            new_move_computers.len(),
        );
        prob = f64::exp((new_score - score) as f64 / temp);
        if score < new_score || rng.gen_bool(prob) {
            score = new_score;
            move_computers = new_move_computers;
            connect_computers = new_connect;
        }

        if best_score < score {
            best_score = score;
            best_move = move_computers.clone();
            best_connect = connect_computers.clone();
        }
    }
    eprintln!("best_score: {}", best_score);
    let move_computers = {
        let mut mv = vec![];
        let grid = start_grid;
        let computers = start_computers;
        for (com_i, next) in best_move {
            let pos = computers[com_i].posi;
            mv.push((pos.0, pos.1, next.0, next.1));
            computers[com_i].go(input, next, grid);
        }
        mv
    };
    (move_computers, best_connect)
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
