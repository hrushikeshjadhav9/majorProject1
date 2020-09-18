        total_step = 0
        episode_rewards = {id_:[] for id_ in intersection_id}
        episode_scores = {id_:[] for id_ in intersection_id}
        with tqdm(total=EPISODES*args.num_step) as pbar:
            for i in range(EPISODES):
                # print("episode: {}".format(i))
                env.reset()
                state = env.get_state()

                episode_length = 0
                episode_reward = {id_:0 for id_ in intersection_id} # for every agent
                episode_score = {id_:0 for id_ in intersection_id} # for everg agent
                while episode_length < args.num_step:
                    
                    action = Magent.choose_action(state) # index of action
                    action_phase = {}
                    for id_, a in action.items():
                        action_phase[id_] = phase_list[id_][a]
                    
                    next_state, reward = env.step(action_phase) # one step
                    score = env.get_score()

                    # consistent time of every phase
                    for _ in range(args.phase_step-1):
                        next_state, reward_ = env.step(action_phase)
                        score_ = env.get_score()
                        for id_ in intersection_id:
                            reward[id_] += reward_[id_]
                            score[id_] += score_[id_]

                    for id_ in intersection_id:
                        reward[id_] /= args.phase_step
                        score[id_] /= args.phase_step

                    for id_ in intersection_id:
                        episode_reward[id_] += reward[id_]
                        episode_score[id_] += score[id_]

                    episode_length += 1
                    total_step += 1
                    pbar.update(1)

                    # store to replay buffer
                    if episode_length > learning_start:
                        Magent.remember(state, action_phase, reward, next_state)

                    state = next_state

                    # training
                    if episode_length > learning_start and total_step % update_model_freq == 0 :
                        if len(Magent.agents[intersection_id[0]].memory) > args.batch_size:
                            Magent.replay()

                    # update target Q netwark
                    if episode_length > learning_start and total_step % update_target_model_freq == 0 :
                        Magent.update_target_network()

                    # logging.info("\repisode:{}/{}, total_step:{}, action:{}, reward:{}"
                    #             .format(i+1, EPISODES, total_step, action, reward))
                    print_reward = {'_'.join(k.split('_')[1:]):v for k, v in reward.items()}
                    pbar.set_description(
                        "t_st:{}, epi:{}, st:{}, r:{}".format(total_step, i+1, episode_length, print_reward))

                # compute episode mean reward
                for id_ in intersection_id:
                    episode_reward[id_] /= args.num_step
                
                # save episode rewards
                for id_ in intersection_id:
                    episode_rewards[id_].append(episode_reward[id_])
                    episode_scores[id_].append(episode_score[id_])
                
                print_episode_reward = {'_'.join(k.split('_')[1:]):v for k, v in episode_reward.items()}
                print_episode_score = {'_'.join(k.split('_')[1:]):v for k, v in episode_score.items()}
                print('\n')
                print("Episode:{}, Mean reward:{}, Score: {}".format(i+1, print_episode_reward, print_episode_score))

                # save model
                if (i + 1) % args.save_freq == 0:
                    if args.algo == 'MDQN':
                        # Magent.save(model_dir + "/{}-ckpt".format(args.algo), i+1)
                        Magent.save(model_dir + "/{}-{}.h5".format(args.algo, i+1))
                        
                    # save reward to file
                    df = pd.DataFrame(episode_rewards)
                    df.to_csv(result_dir + '/rewards.csv', index=None)

                    df = pd.DataFrame(episode_scores)
                    df.to_csv(result_dir + '/scores.csv', index=None)

                    # save figure
                    plot_data_lists([episode_rewards[id_] for id_ in intersection_id], intersection_id, figure_name=result_dir + '/rewards.pdf')
                    plot_data_lists([episode_scores[id_] for id_ in intersection_id], intersection_id, figure_name=result_dir + '/scores.pdf')
        