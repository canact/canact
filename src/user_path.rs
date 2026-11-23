//! Expand a `~` path without joining a remainder that leaves home.

use std::path::{Component, Path, PathBuf};

/// Expand a leading `~`, `~/`, or `~\` onto `home`.
///
/// A remainder that is a root, a drive, or a `..` that climbs past
/// `home` is returned unchanged. A `..` that stays inside home is
/// joined (`~/skills/../skills`). `\` is counted both as a separator
/// and as part of one name, so the answer is the same on Windows and
/// Unix. `~user` is unchanged. `home == None` leaves a tilde token
/// unchanged.
pub fn expand_tilde_with_home(path: &str, home: Option<&Path>) -> PathBuf {
    let Some(rest) = tilde_remainder(path) else {
        return PathBuf::from(path);
    };
    let Some(home) = home else {
        return PathBuf::from(path);
    };
    if rest.is_empty() {
        return home.to_path_buf();
    }
    if remainder_replaces_base(rest) {
        return PathBuf::from(path);
    }
    home.join(rest)
}

fn tilde_remainder(path: &str) -> Option<&str> {
    let after = path.strip_prefix('~')?;
    if after.is_empty() {
        return Some("");
    }
    if !after.starts_with('/') && !after.starts_with('\\') {
        return None;
    }
    Some(after.trim_start_matches(['/', '\\']))
}

fn remainder_replaces_base(rest: &str) -> bool {
    let path = Path::new(rest);
    if path.has_root()
        || path
            .components()
            .any(|component| matches!(component, Component::Prefix(_)))
        || looks_like_windows_drive(rest)
    {
        return true;
    }
    // Unix `Path` keeps `\` in one component, so `..\..\Windows` would
    // join under home here and leave home on Windows. A name that
    // contains `\` (`a\b/../../outside`) is one directory on Unix and
    // two on Windows. Refuse when either walk climbs past home.
    rest_climbs_past_base(rest)
}

fn rest_climbs_past_base(rest: &str) -> bool {
    climb_past_base(rest.split(['/', '\\'])) || climb_past_base(rest.split('/'))
}

fn climb_past_base<'a>(parts: impl Iterator<Item = &'a str>) -> bool {
    let mut depth: u32 = 0;
    for component in parts {
        if component.is_empty() || component == "." {
            continue;
        }
        if component == ".." {
            if depth == 0 {
                return true;
            }
            depth -= 1;
        } else {
            depth = depth.saturating_add(1);
        }
    }
    false
}

fn looks_like_windows_drive(rest: &str) -> bool {
    let mut chars = rest.chars();
    let Some(letter) = chars.next() else {
        return false;
    };
    letter.is_ascii_alphabetic() && chars.next() == Some(':')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parent_inside_home_stays_joined() {
        let home = Path::new("/tmp/iso-home");
        assert_eq!(
            expand_tilde_with_home("~/skills/../skills", Some(home)),
            home.join("skills/../skills")
        );
    }

    #[test]
    fn parent_that_leaves_home_is_unchanged() {
        let home = Path::new("/tmp/iso-home");
        for raw in [
            "~/../../Windows",
            r"~\..\..\Windows",
            "~/a\\b/../../outside",
            "~//C:/Windows",
        ] {
            assert_eq!(
                expand_tilde_with_home(raw, Some(home)),
                PathBuf::from(raw),
                "{raw}"
            );
        }
    }

    #[test]
    fn bare_tilde_and_plain_paths() {
        let home = Path::new("/tmp/iso-home");
        assert_eq!(expand_tilde_with_home("~", Some(home)), home);
        assert_eq!(
            expand_tilde_with_home("~user/path", Some(home)),
            PathBuf::from("~user/path")
        );
        assert_eq!(
            expand_tilde_with_home("/tmp/overlays", Some(home)),
            PathBuf::from("/tmp/overlays")
        );
        assert_eq!(
            expand_tilde_with_home("~/overlays", None),
            PathBuf::from("~/overlays")
        );
    }
}
