def test(agent, environment, img_processing, cfg):

    # Counters 
    wall_collisions = 0
    box_collisions = 0
    endless_loops = 0
    step_count = 0
    cumulative_reward = 0
    side_steps = 0 # Side-step counter (to prevent endless loops)


    # Reset environment at start of episode
    _, _, _ = environment.reset(cfg['training_condition'])

    # Create an empty frame stack and fill it with frames
    frame_stack = utils.FrameStack(stack_size=cfg['stack_size'] )
    for _ in range(cfg['stack_size'] ):
        _, _, frame_raw = environment.step(0)
        frame = img_processing(frame_raw).to(agent.device) 
        state = frame_stack.update_with(frame)

    # Episode starts here:
    agent_finished = False
    while not agent_finished:

        # 1. Agent performs a step (based on the current state) and obtains next state
        action = agent.select_action(state, validation=True)
        side_steps = side_steps + 1  if action != 0 else 0
        if side_steps > cfg['reset_after_nr_sidesteps']:
            endless_loops +=1
            action = torch.zeros_like(action) # force forward step
            side_steps = 0
        
        end, reward, frame_raw = environment.step(action.item())
        agent_finished = cfg['reset_upon_end_signal'][end]
        frame = img_processing(frame_raw).to(agent.device)
        next_state = frame_stack.update_with(frame) if not agent_finished else None      

        # 2. Interpret reward signal
        if reward > 100:
            reward = -(reward -100)

        # 3. push to replay memory (not used by model, for evaluation only) 
        reward = torch.tensor([reward], device=agent.device,dtype=torch.float)
        action = action.unsqueeze(0)
        agent.memory.push(state, action, next_state, reward)    
            
            
        # 4. Store performance and training measures
        step_count += 1
        cumulative_reward += reward;
        if end == 1:
            box_collisions += 1
        if end == 2:
            wall_collisions +=1 
            
        # 5. Go to next iteration    
        state = next_state
    
    return {'wall_collisions':wall_collisions, 
            'box_collisions' :box_collisions,
            'endless_loops' :endless_loops,
            'step_count' :step_count,
            'cumulative_reward' :cumulative_reward}