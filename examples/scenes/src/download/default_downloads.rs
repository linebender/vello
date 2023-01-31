// This content cannot be formatted by rustfmt because of the long strings, so it's in its own file
use super::{BuiltinSvgProps, SVGDownload};

pub(super) fn default_downloads() -> Vec<SVGDownload> {
    vec![
        SVGDownload {
            builtin:Some(BuiltinSvgProps {
                info: "https://commons.wikimedia.org/wiki/File:CIA_WorldFactBook-Political_world.svg",
                license: "Public Domain",
                expected_size: 12771150,
            }),
            url: "https://upload.wikimedia.org/wikipedia/commons/7/72/Political_Map_of_the_World_%28august_2013%29.svg".to_string(),
            name: "CIA World Map".to_string()
        },
        SVGDownload {
            builtin:Some(BuiltinSvgProps {
                info: "https://commons.wikimedia.org/wiki/File:World_-_time_zones_map_(2014).svg",
                license: "Public Domain",
                expected_size: 5235172,
            }),
            url: "https://upload.wikimedia.org/wikipedia/commons/c/c6/World_-_time_zones_map_%282014%29.svg".to_string(),
            name: "Time Zones Map".to_string()
        },
        SVGDownload {
            builtin:Some(BuiltinSvgProps {
                info: "https://commons.wikimedia.org/wiki/File:Coat_of_arms_of_Poland-official.svg",
                license: "Public Domain",
                expected_size: 10747708,
            }),
            url: "https://upload.wikimedia.org/wikipedia/commons/3/3e/Coat_of_arms_of_Poland-official.svg".to_string(),
            name: "Coat of Arms of Poland".to_string()
        },
        SVGDownload {
            builtin:Some(BuiltinSvgProps {
                info: "https://commons.wikimedia.org/wiki/File:Coat_of_arms_of_the_Kingdom_of_Yugoslavia.svg",
                license: "Public Domain",
                expected_size: 12795806,
            }),
            url: "https://upload.wikimedia.org/wikipedia/commons/5/58/Coat_of_arms_of_the_Kingdom_of_Yugoslavia.svg".to_string(),
            name: "Coat of Arms of the Kingdom of Yugoslavia".to_string()
        },
    ]
}
