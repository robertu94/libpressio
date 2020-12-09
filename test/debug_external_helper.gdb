# pagination breaks scripts because it non-deterministicly adds extra command prompts
set pagination off

# start the binary to load dynamically loaded libraries, if needed
start

# install the fork hook after gathering defaults
break external_metric_plugin::build_command
commands
  # follow the child process
  set follow-fork-mode child
  set follow-exec-mode new

  # break on main in the child
  break main

  # keep going until we hit main in the child process
  continue
end

continue

# vim: set ft=gdb :
