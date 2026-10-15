//! A run's account, the same in both games: the score and its multiplier, the lives and bombs
//! left, and the extras the score earns.

/// The score of the first extra life, and between the next ones.
pub const LIFE_EVERY: u64 = 100_000;
/// The score of the first extra bomb, and between the next ones.
pub const BOMB_EVERY: u64 = 150_000;

/// The score, multiplier, lives and bombs of a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ledger {
    /// Score.
    pub score: u64,
    /// Multiplier.
    pub mult: u32,
    /// Lives left, including the current one.
    pub lives: u32,
    /// Bombs left.
    pub bombs: u32,
    /// The score of the next extra life.
    next_life: u64,
    /// The score of the next extra bomb.
    next_bomb: u64,
}

impl Default for Ledger {
    fn default() -> Ledger {
        Ledger::new()
    }
}

impl Ledger {
    /// A run's start: no score, a multiplier of one, three lives and three bombs.
    pub fn new() -> Ledger {
        Ledger {
            score: 0,
            mult: 1,
            lives: 3,
            bombs: 3,
            next_life: LIFE_EVERY,
            next_bomb: BOMB_EVERY,
        }
    }

    /// Add `points` to the score; returns them.
    pub fn award(&mut self, points: u64) -> u64 {
        self.score += points;
        points
    }

    /// The multiplier up by one (a shard collected, a gate flown through); returns it.
    pub fn raise(&mut self) -> u32 {
        self.mult += 1;
        self.mult
    }

    /// Use a bomb, if one is left.
    pub fn take_bomb(&mut self) -> bool {
        let any = self.bombs > 0;
        self.bombs = self.bombs.saturating_sub(1);
        any
    }

    /// A ship lost: a life and the multiplier go. Returns whether a life is left.
    pub fn lose_life(&mut self) -> bool {
        self.mult = 1;
        self.lives = self.lives.saturating_sub(1);
        self.lives > 0
    }

    /// An extra life every [`LIFE_EVERY`] points and an extra bomb every [`BOMB_EVERY`], as
    /// many as the score has passed (a big kill may pass more than one): `earned(true)` for each
    /// life, then `earned(false)` for each bomb.
    pub fn extras(&mut self, mut earned: impl FnMut(bool)) {
        while self.score >= self.next_life {
            self.next_life += LIFE_EVERY;
            self.lives += 1;
            earned(true);
        }
        while self.score >= self.next_bomb {
            self.next_bomb += BOMB_EVERY;
            self.bombs += 1;
            earned(false);
        }
    }

    /// The scores of the next extra life and bomb (for the replays' state hash).
    pub fn thresholds(&self) -> [u64; 2] {
        [self.next_life, self.next_bomb]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A score passing several thresholds at once earns them all, lives before bombs.
    #[test]
    fn extras_come_at_their_thresholds() {
        let mut l = Ledger::new();
        let mut got = Vec::new();
        l.award(LIFE_EVERY - 1);
        l.extras(|life| got.push(life));
        assert!(got.is_empty());
        l.award(2 * BOMB_EVERY);
        l.extras(|life| got.push(life));
        assert_eq!(got, [true, true, true, false, false]);
        assert_eq!((l.lives, l.bombs), (6, 5));
        assert_eq!(l.thresholds(), [4 * LIFE_EVERY, 3 * BOMB_EVERY]);
    }

    /// A lost life resets the multiplier; the last one says so.
    #[test]
    fn lives_and_bombs_run_out() {
        let mut l = Ledger::new();
        assert_eq!(l.raise(), 2);
        assert!(l.lose_life() && l.mult == 1);
        assert!(l.lose_life());
        assert!(!l.lose_life());
        assert!(l.take_bomb() && l.take_bomb() && l.take_bomb());
        assert!(!l.take_bomb());
        assert_eq!(l.bombs, 0);
    }
}
