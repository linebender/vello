#!/bin/bash

# If there are new files with headers that can't match the conditions here,
# then the files can be ignored by an additional glob argument via the -g flag.
# For example:
#   -g "!src/special_file.rs"
#   -g "!src/special_directory"

# Check all the standard Rust source files
output=$(rg "^// Copyright (19|20)[\d]{2} (.+ and )?the Vello Authors( and .+)?$\n^// SPDX-License-Identifier: Apache-2\.0 OR MIT$\n\n" --files-without-match --multiline -g "*.rs" -g "!vello_shaders/{shader,src/cpu}" .)

if [ -n "$output" ]; then
	echo -e "The following files lack the correct copyright header:\n"
	echo $output
	echo -e "\n\nPlease add the following header:\n"
	echo "// Copyright $(date +%Y) the Vello Authors"
	echo "// SPDX-License-Identifier: Apache-2.0 OR MIT"
	echo -e "\n... rest of the file ...\n"
	exit 1
fi

# Check all the shaders, both WGSL and CPU shaders in Rust, as they also have Unlicense
output=$(rg "^// Copyright (19|20)[\d]{2} (.+ and )?the Vello Authors( and .+)?$\n^// SPDX-License-Identifier: Apache-2\.0 OR MIT OR Unlicense$\n\n" --files-without-match --multiline -g "vello_shaders/{shader,src/cpu}/**/*.{rs,wgsl}" .)

if [ -n "$output" ]; then
        echo -e "The following shader files lack the correct copyright header:\n"
        echo $output
        echo -e "\n\nPlease add the following header:\n"
        echo "// Copyright $(date +%Y) the Vello Authors"
        echo "// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense"
        echo -e "\n... rest of the file ...\n"
        exit 1
fi

echo "All files have correct copyright headers."
exit 0

