use rand::prelude::*;

const TIMELIMIT: f64 = 2.725;
const DIJ: [(usize, usize); 4] = [(0, !0), (!0, 0), (0, 1), (1, 0)];
// const DIR: [char; 4] = ['L', 'U', 'R', 'D'];
type Output = Vec<(usize, usize, usize, usize)>;

fn main() {
    let timer = Timer::new();
    let input = Input::new();
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(356296);
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
                        connect: vec![],
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
    const T0: f64 = 25.0;
    const T1: f64 = 0.01;
    let mut temp = T0;
    let mut prob;

    let mut moves: Vec<(usize, (usize, usize), (usize, usize))> = vec![];
    let mut connects: Vec<(usize, usize, usize, usize)> = vec![];

    let mut score = 0;
    let mut best_score = score;
    let mut best_moves = vec![];
    let mut best_connects = vec![];
    // movesの順にcomputerを動かす
    // どの向きに動かすかは探索する（今はランダム）
    let mut count = 0;
    // let mut neigh0 = 0;
    // let mut neigh1 = 0;

    let mut grid = start_grid.to_owned();
    let mut computers = start_computers.to_owned();
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
        // 近傍解作成
        let move_neigh_count = 2;
        let move_neigh = rng.gen_range(0, move_neigh_count);
        match move_neigh {
            0 => {
                // insert
                if 100 * input.k <= new_moves.len() {
                    continue 'lp;
                }
                let insert_i = if new_moves.is_empty() {
                    0
                } else {
                    rng.gen_range(0, new_moves.len())
                };
                let mut new = vec![];
                for &(com_i, prev, next) in new_moves.iter().take(insert_i) {
                    new.push((com_i, prev, next));
                }
                for &(com_i, prev, next) in new_moves.iter().skip(insert_i).rev() {
                    if grid[prev.0][prev.1].is_cable() {
                        // cableを削除する
                        if prev.0 == next.0 && grid[prev.0][prev.1].is_vertical() {
                            // 横に移動して、cableが縦なら繋がっているところも削除
                            for (_, &(di, dj)) in
                                DIJ.iter().enumerate().filter(|(dir, _)| *dir & 1 == 1)
                            {
                                let mut ni = prev.0;
                                let mut nj = prev.1;
                                for _ in 1..input.n {
                                    ni += di;
                                    nj += dj;
                                    if grid[ni][nj].is_cable() {
                                        grid[ni][nj] = Cell::Empty;
                                    } else {
                                        break;
                                    }
                                }
                            }
                        }
                        if prev.1 == next.1 && grid[prev.0][prev.1].is_horizon() {
                            // 縦に移動して、cableが横なら繋がっているところも削除
                            for (_, &(di, dj)) in
                                DIJ.iter().enumerate().filter(|(dir, _)| *dir & 1 == 0)
                            {
                                let mut ni = prev.0;
                                let mut nj = prev.1;
                                for _ in 1..input.n {
                                    ni += di;
                                    nj += dj;
                                    if grid[ni][nj].is_cable() {
                                        grid[ni][nj] = Cell::Empty;
                                    } else {
                                        break;
                                    }
                                }
                            }
                        }
                        grid[prev.0][prev.1] = Cell::Empty;
                    }
                    if !computers[com_i].go(input, prev, &mut grid) {
                        unreachable!();
                    }
                }
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
                            Cell::Computer { index: _ } => {}
                            Cell::Cable { kind, dir: _ } => {
                                if computer.kind == kind {
                                    moveable.push((i, (ni, nj)));
                                }
                            }
                        }
                    }
                }
                for _ in 0..rng.gen_range(1, 6) {
                    if moveable.is_empty() {
                        break;
                    }
                    let (com_i, next) = moveable[rng.gen_range(0, moveable.len())];
                    let prev = computers[com_i].posi;
                    // cableに移動するときは、nextのcableを削除する
                    if grid[next.0][next.1].is_cable() {
                        // cableを削除する
                        if prev.0 == next.0 && grid[next.0][next.1].is_vertical() {
                            // 横に移動して、cableが縦なら繋がっているところも削除
                            for (_, &(di, dj)) in
                                DIJ.iter().enumerate().filter(|(dir, _)| *dir & 1 == 1)
                            {
                                let mut ni = next.0;
                                let mut nj = next.1;
                                for _ in 1..input.n {
                                    ni += di;
                                    nj += dj;
                                    if grid[ni][nj].is_cable() {
                                        grid[ni][nj] = Cell::Empty;
                                    } else {
                                        break;
                                    }
                                }
                            }
                        }
                        if prev.1 == next.1 && grid[next.0][next.1].is_horizon() {
                            // 縦に移動して、cableが横なら繋がっているところも削除
                            for (_, &(di, dj)) in
                                DIJ.iter().enumerate().filter(|(dir, _)| *dir & 1 == 0)
                            {
                                let mut ni = next.0;
                                let mut nj = next.1;
                                for _ in 1..input.n {
                                    ni += di;
                                    nj += dj;
                                    if grid[ni][nj].is_cable() {
                                        grid[ni][nj] = Cell::Empty;
                                    } else {
                                        break;
                                    }
                                }
                            }
                        }
                        grid[next.0][next.1] = Cell::Empty;
                    }
                    if !computers[com_i].go(input, next, &mut grid) {
                        unreachable!();
                    }
                    new.push((com_i, prev, next));
                    // moveableを変化させる
                    let mut new_moveable = vec![];
                    // moveable.push((com_i, prev)); // 戻すような操作は加えない
                    // com_iがprev以外の方向に動かせるかは調べる
                    for &(di, dj) in DIJ.iter() {
                        let ni = next.0 + di;
                        let nj = next.1 + dj;
                        if input.n <= ni || input.n <= nj || (ni, nj) == prev {
                            continue;
                        }
                        if grid[ni][nj].is_empty() {
                            new_moveable.push((com_i, (ni, nj)));
                        }
                        if grid[ni][nj].is_cable() {
                            let kind = grid[ni][nj].kind();
                            if computers[com_i].kind == kind {
                                new_moveable.push((com_i, (ni, nj)));
                            }
                        }
                    }
                    // prevから4近傍を見る
                    for &(di, dj) in DIJ.iter() {
                        let ni = prev.0 + di;
                        let nj = prev.1 + dj;
                        if input.n <= ni || input.n <= nj || (ni, nj) == next {
                            continue;
                        }
                        // prevのところが空きマスになるので、(ni, nj)にComputerがあればmoveableになる
                        if let Cell::Computer { index } = grid[ni][nj] {
                            new_moveable.push((index, prev));
                        }
                    }
                    // old_moveableで移動先がnextだったやつと、m.0 == com_iは不採用
                    for m in moveable {
                        if m.1 == next || m.0 == com_i {
                            continue;
                        }
                        new_moveable.push(m);
                    }
                    moveable = new_moveable;
                }
                for &(com_i, prev, next) in new_moves.iter().skip(insert_i) {
                    if computers[com_i].go(input, next, &mut grid) {
                        new.push((com_i, prev, next));
                    }
                }
                new_moves = new;
            }
            1 => {
                // remove
                let mut new_grid = start_grid.to_owned();
                let mut new_computers = start_computers.to_owned();
                for _ in 0..rng.gen_range(1, input.k * 2) {
                    if new_moves.is_empty() {
                        break;
                    }
                    let i = rng.gen_range(0, new_moves.len());
                    new_moves.remove(i);
                }
                for &(com_i, _, next) in new_moves.iter() {
                    if !new_computers[com_i].go(input, next, &mut new_grid) {
                        continue 'lp;
                    }
                }
                grid = new_grid;
                computers = new_computers;
            }
            _ => unreachable!(),
        }

        let mut new_connects = vec![];
        let mut uf = UnionFind::new(input.n * input.n);
        'connect_lp: for id1 in 0..computers.len() {
            let i = computers[id1].posi.0;
            let j = computers[id1].posi.1;
            for (dir, &(di, dj)) in DIJ.iter().enumerate() {
                // TODO: new_computers[id1].connectをチェックしてこの部分を高速化
                let mut ni = i;
                let mut nj = j;
                for len in 1..input.n {
                    ni += di;
                    nj += dj;
                    if input.n <= ni || input.n <= nj {
                        break;
                    }
                    if grid[ni][nj].is_empty() {
                        continue;
                    }
                    if grid[ni][nj].is_cable() {
                        break;
                    }
                    if uf.same(i * input.n + j, ni * input.n + nj) {
                        break;
                    }
                    let id2 = grid[ni][nj].index();
                    if computers[id1].kind == computers[id2].kind {
                        // new_computers[id1].connect.push(id2);
                        // new_computers[id2].connect.push(id1);
                        new_connects.push((i, j, ni, nj));
                        uf.unite(i * input.n + j, ni * input.n + nj);
                        for _ in 0..len - 1 {
                            ni -= di;
                            nj -= dj;
                            grid[ni][nj] = Cell::Cable {
                                kind: computers[id1].kind,
                                dir,
                            };
                        }
                    }
                    if 100 * input.k <= new_moves.len() + new_connects.len() {
                        break 'connect_lp;
                    }
                    break;
                }
            }
        }

        // 近傍解作成ここまで
        let new_score = compute_score(input, &computers, &mut uf);
        prob = f64::exp((new_score - score) as f64 / temp);
        if score < new_score || rng.gen_bool(prob) {
            score = new_score;
            moves = new_moves;
            connects = new_connects;
        } else {
            for &(com_i, prev, next) in new_moves.iter().rev() {
                if grid[prev.0][prev.1].is_cable() {
                    // cableを削除する
                    if prev.0 == next.0 && grid[prev.0][prev.1].is_vertical() {
                        // 横に移動して、cableが縦なら繋がっているところも削除
                        for (_, &(di, dj)) in
                            DIJ.iter().enumerate().filter(|(dir, _)| *dir & 1 == 1)
                        {
                            let mut ni = prev.0;
                            let mut nj = prev.1;
                            for _ in 1..input.n {
                                ni += di;
                                nj += dj;
                                if grid[ni][nj].is_cable() {
                                    grid[ni][nj] = Cell::Empty;
                                } else {
                                    break;
                                }
                            }
                        }
                    }
                    if prev.1 == next.1 && grid[prev.0][prev.1].is_horizon() {
                        // 縦に移動して、cableが横なら繋がっているところも削除
                        for (_, &(di, dj)) in
                            DIJ.iter().enumerate().filter(|(dir, _)| *dir & 1 == 0)
                        {
                            let mut ni = prev.0;
                            let mut nj = prev.1;
                            for _ in 1..input.n {
                                ni += di;
                                nj += dj;
                                if grid[ni][nj].is_cable() {
                                    grid[ni][nj] = Cell::Empty;
                                } else {
                                    break;
                                }
                            }
                        }
                    }
                    grid[prev.0][prev.1] = Cell::Empty;
                }
                if !computers[com_i].go(input, prev, &mut grid) {
                    unreachable!();
                }
            }
            for &(com_i, prev, next) in moves.iter() {
                // cableに移動するときは、nextのcableを削除する
                if grid[next.0][next.1].is_cable() {
                    // cableを削除する
                    if prev.0 == next.0 && grid[next.0][next.1].is_vertical() {
                        // 横に移動して、cableが縦なら繋がっているところも削除
                        for (_, &(di, dj)) in
                            DIJ.iter().enumerate().filter(|(dir, _)| *dir & 1 == 1)
                        {
                            let mut ni = next.0;
                            let mut nj = next.1;
                            for _ in 1..input.n {
                                ni += di;
                                nj += dj;
                                if grid[ni][nj].is_cable() {
                                    grid[ni][nj] = Cell::Empty;
                                } else {
                                    break;
                                }
                            }
                        }
                    }
                    if prev.1 == next.1 && grid[next.0][next.1].is_horizon() {
                        // 縦に移動して、cableが横なら繋がっているところも削除
                        for (_, &(di, dj)) in
                            DIJ.iter().enumerate().filter(|(dir, _)| *dir & 1 == 0)
                        {
                            let mut ni = next.0;
                            let mut nj = next.1;
                            for _ in 1..input.n {
                                ni += di;
                                nj += dj;
                                if grid[ni][nj].is_cable() {
                                    grid[ni][nj] = Cell::Empty;
                                } else {
                                    break;
                                }
                            }
                        }
                    }
                    grid[next.0][next.1] = Cell::Empty;
                }
                if !computers[com_i].go(input, next, &mut grid) {
                    unreachable!();
                }
            }
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
        for (com_i, prev, next) in best_moves {
            mv.push((prev.0, prev.1, next.0, next.1));
            computers[com_i].go(input, next, grid);
        }
        mv
    };
    (moves, best_connects)
}

fn compute_score(input: &Input, computers: &[Computer], uf: &mut UnionFind) -> i64 {
    let mut score = 0;
    for (i, computer1) in computers.iter().enumerate() {
        for computer2 in computers.iter().skip(i + 1) {
            if uf.same(
                computer1.posi.0 * input.n + computer1.posi.1,
                computer2.posi.0 * input.n + computer2.posi.1,
            ) {
                if computer1.kind == computer2.kind {
                    score += 1;
                } else {
                    score -= 1;
                }
            }
        }
    }
    score
}

#[allow(dead_code)]
fn dist(p1: (usize, usize), p2: (usize, usize)) -> usize {
    let dx = if p1.0 > p2.0 {
        p1.0 - p2.0
    } else {
        p2.0 - p1.0
    };
    let dy = if p1.1 > p2.1 {
        p1.1 - p2.1
    } else {
        p2.1 - p1.1
    };
    dx * dx + dy * dy
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Cell {
    Empty,
    Computer { index: usize },
    Cable { kind: usize, dir: usize },
}

impl Cell {
    fn is_empty(&self) -> bool {
        matches!(*self, Cell::Empty)
    }

    fn is_cable(&self) -> bool {
        matches!(*self, Cell::Cable { kind: _, dir: _ })
    }

    fn index(&self) -> usize {
        if let Cell::Computer { index } = self {
            *index
        } else {
            panic!("cell is not computer");
        }
    }

    fn dir(&self) -> usize {
        if let Cell::Cable { kind: _, dir } = self {
            *dir
        } else {
            panic!("cell is not cable");
        }
    }

    fn kind(&self) -> usize {
        if let Cell::Cable { kind, dir: _ } = self {
            *kind
        } else {
            panic!("cell is not cable");
        }
    }

    fn is_vertical(&self) -> bool {
        let dir = self.dir();
        dir & 1 == 1
    }

    fn is_horizon(&self) -> bool {
        !self.is_vertical()
    }
}

#[derive(Clone)]
struct Computer {
    posi: (usize, usize),
    kind: usize,
    connect: Vec<usize>, // computer_index
}

impl Computer {
    fn go(&mut self, input: &Input, next: (usize, usize), grid: &mut [Vec<Cell>]) -> bool {
        if !grid[next.0][next.1].is_empty() {
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
