Experiment process for finding UCT C hyper parameter value for I-NTMCP
######################################################################

1. Determine r_hi:
   - Run NST l=0 with c=0 for different number of simulations against a random opponent
     - NST should be using random rollout policy
   - Run for a smallish number of episode e.g. 10-50, depending of how variable environment is
   - r_hi is the max score achieved by NST l=0.
     - Use the max over all the different #sims used
2. Determine r_lo:
   - Run random vs random agent
   - Run for same or larger number of episodes as step 1
   - r_lo is the min score achieved by random agent
3. Calculate UCT C
   - C = r_hi - r_lo
4. Validate
   - Test NST l=0 against random using uct c =0 vs handpicked c value vs calculated c value
     - Test using No rollout policy
     - Test using rollout policy where:
       - v_init set to r_hi for prefered actions and r_lo for unpreferred actions
       - n_init set to 10 for prefered actions and 0 for unpreferred actions
