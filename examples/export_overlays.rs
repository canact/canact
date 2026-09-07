//! Write Aider and Cline overlay files from an in-memory profile.
//!
//! No network and no API key. Pass an output directory, or files go
//! under a `canact-overlays` folder in the process temp directory.

include!("include/sample_profile.rs");

use std::env;
use std::path::PathBuf;

use canact::HostOverlay;

fn main() {
    let dir = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| env::temp_dir().join("canact-overlays"));
    let profile = sample_profile();
    let advertised = Some(40_960);
    HostOverlay::aider(&profile, advertised)
        .write_to(&dir)
        .expect("write Aider overlay");
    HostOverlay::cline(&profile, advertised)
        .write_to(&dir)
        .expect("write Cline overlay");
    for name in [
        ".aider.model.settings.yml",
        ".aider.model.metadata.json",
        "cline.modelinfo.json",
    ] {
        println!("{}", dir.join(name).display());
    }
}
