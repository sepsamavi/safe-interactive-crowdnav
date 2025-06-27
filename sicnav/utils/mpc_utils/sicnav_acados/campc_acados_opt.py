from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import numpy as np
import casadi as cs
import os

def export_campc_model(mpc_env):

    model_name = 'sicnav_mpc' if mpc_env.hum_model == 'orca_casadi_kkt' else 'mpc_cvmm'

    horiz = mpc_env.horiz
    nx = mpc_env.nx
    nu = mpc_env.nu


    model = AcadosModel()

    # Define model variables
    # ---------------------
    model.x = mpc_env.system_model.x_sym
    model.u = mpc_env.system_model.u_sym

    if mpc_env.hum_model == 'orca_casadi_kkt' and mpc_env.human_pred_MID:
        stacked_preds = mpc_env.stack_MID_preds(mpc_env.MID_samples_t_all_hums_stacked, mpc_env.MID_samples_tp1_all_hums_stacked)

        if not mpc_env.outdoor_robot_setting:
            cost_params = cs.vertcat(mpc_env.system_model.Xr, mpc_env.system_model.Ur, mpc_env.system_model.Q_diag, mpc_env.system_model.R_diag, mpc_env.system_model.term_Q_diag, stacked_preds)
        else:
            cost_params = cs.vertcat(mpc_env.system_model.Xr, mpc_env.system_model.Ur, mpc_env.system_model.Q_diag, mpc_env.system_model.R_diag, mpc_env.system_model.term_Q_diag, stacked_preds, mpc_env.stat_obs_params_vecced)
    else:
        if not mpc_env.outdoor_robot_setting:
            cost_params = cs.vertcat(mpc_env.system_model.Xr, mpc_env.system_model.Ur, mpc_env.system_model.Q_diag, mpc_env.system_model.R_diag, mpc_env.system_model.term_Q_diag)
        else:
            # stat_obs_params_properly_reshaped = cs.horzcat(*tuple([mpc_env.stat_obs_params[idx,:] for idx in range(mpc_env.num_stat_obs)])).T
            cost_params = cs.vertcat(mpc_env.system_model.Xr, mpc_env.system_model.Ur, mpc_env.system_model.Q_diag, mpc_env.system_model.R_diag, mpc_env.system_model.term_Q_diag, mpc_env.stat_obs_params_vecced)

    model.p = cost_params


    if mpc_env.hum_model == 'orca_casadi_kkt' and mpc_env.human_pred_MID:
        model.disc_dyn_expr = mpc_env.system_model.f_func_nonlin(mpc_env.system_model.x_sym, mpc_env.system_model.u_sym, stacked_preds)
    else:
        model.disc_dyn_expr = mpc_env.system_model.f_func_nonlin(mpc_env.system_model.x_sym, mpc_env.system_model.u_sym)

    model.name = model_name

    return model

def export_campc_ocp(mpc_env, file_run_id=None):


    hash = mpc_env.config_hash
    code_export_dir = 'acados_cache/c_generated_code_{:}'.format(hash)
    json_file = 'acados_cache/acados_sicnav_mpc_{:}.json'.format(hash)
    # code_export_dir = 'c_generated_code_0' if file_run_id is None else 'c_generated_code_{:}'.format(file_run_id)
    # json_file = 'acados_sicnav_mpc.json' if file_run_id is None else 'acados_sicnav_mpc_{:}.json'.format(file_run_id)
    regen = not (os.path.exists(code_export_dir) and os.path.isdir(code_export_dir))
    # regen = True

    ocp = AcadosOcp()
    ocp.code_export_directory = code_export_dir
    # -------------------------------------------------------------------------------
    # set model
    model = export_campc_model(mpc_env)
    ocp.model = model
    Tf = mpc_env.horiz * mpc_env.time_step
    nx = mpc_env.nx
    nu = mpc_env.nu
    N = mpc_env.horiz

    ocp.dims.N = N

    # -------------------------------------------------------------------------------
    # set cost
    # Q_mat = mpc_env.Q
    # term_Q_mat = mpc_env.term_Q
    # R_mat = mpc_env.R

    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost = mpc_env.system_model.cost_func(mpc_env.system_model.x_sym, mpc_env.system_model.u_sym, mpc_env.system_model.Xr, mpc_env.system_model.Ur, mpc_env.system_model.Q_diag, mpc_env.system_model.R_diag)

    ocp.model.cost_expr_ext_cost_e = mpc_env.system_model.term_cost_func(mpc_env.system_model.x_sym, mpc_env.system_model.Xr, mpc_env.system_model.term_Q_diag)

    # -------------------------------------------------------------------------------
    # dummy set the parameter values
    ocp.parameter_values = np.zeros((model.p.shape[0],), dtype=np.float64)

    # -------------------------------------------------------------------------------
    #dummy setting the constraints
    x0 = np.zeros((nx,), dtype=np.float64)
    ocp.constraints.x0 = x0

    h_expr_list = []
    h_rownames_list = []
    h_ub_list = []
    h_lb_list = []
    slacked_list = []
    h_e_expr_list = []
    h_e_rownames_list = []
    h_e_ub_list = []
    h_e_lb_list = []
    slacked_e_list = []

    inf_val = 1e6
    h_idxs = []
    h_idx = 0
    # state constraints:
    for sc_i, state_constraint in enumerate(mpc_env.state_constraints):
        # needs to be in the form h(x_t, u_t, p) - Jsh suh_t <= h_upper

        cs_eqn = state_constraint.sym_cs_func(input=mpc_env.system_model.x_sym)['const']
        h_expr_list.append(cs_eqn)
        h_e_expr_list.append(cs_eqn)
        slack_val = 1.0
        print('[OPT] state constraint: {:}, {:}'.format(state_constraint.name, 'slacked' if slack_val != 0 else 'not slacked'))
        for idx in range(cs_eqn.shape[0]):
            h_ub_list.append(0.0)
            h_lb_list.append(-inf_val)
            h_e_ub_list.append(0.0)
            h_e_lb_list.append(-inf_val)
            slacked_list.append(slack_val)
            slacked_e_list.append(slack_val)
            h_rownames_list.append(state_constraint.name+'_r{:}_{:}'.format(idx, state_constraint.row_names[idx]))
            h_e_rownames_list.append(state_constraint.name+'_r{:}_{:}'.format(idx, state_constraint.row_names[idx]))

    # CAUSE NUMERICAL ISSUE with CVMM
    for sc_i, stat_coll_con in enumerate(mpc_env.stat_coll_consts):
        if mpc_env.outdoor_robot_setting:
            cs_eqn = stat_coll_con.sym_cs_func(input=cs.vertcat(mpc_env.system_model.x_sym, mpc_env.stat_obs_params_vecced))['const']
        else:
            cs_eqn = stat_coll_con.sym_cs_func(input=mpc_env.system_model.x_sym)['const']
        # cs_eqn = stat_coll_con.sym_cs_func(input=cs.vertcat(mpc_env.system_model.x_sym, mpc_env.system_model.u_sym))['const']
        h_expr_list.append(cs_eqn)
        h_e_expr_list.append(cs_eqn)
        slack_val = 1.0
        print('[OPT] state constraint: {:}, {:}'.format(stat_coll_con.name, 'slacked' if slack_val != 0.0 else 'not slacked'))
        for idx in range(cs_eqn.shape[0]):
            h_ub_list.append(0.0)
            h_lb_list.append(-inf_val)
            h_e_ub_list.append(0.0)
            h_e_lb_list.append(-inf_val)
            slacked_list.append(slack_val)
            slacked_e_list.append(slack_val)
            h_rownames_list.append(stat_coll_con.name+'_r{:}_{:}'.format(idx, stat_coll_con.row_names[idx]))
            h_e_rownames_list.append(stat_coll_con.name+'_r{:}_{:}'.format(idx, stat_coll_con.row_names[idx]))

    for ic_i, input_constraint in enumerate(mpc_env.input_constraints):
        cs_eqn = input_constraint.sym_cs_func(input=mpc_env.system_model.u_sym)['const']
        h_expr_list.append(cs_eqn)
        slack_val = 0.0
        print('[OPT] input constraint: {:}, {:}'.format(input_constraint.name, 'slacked' if slack_val != 0.0 else 'not slacked'))
        for idx in range(cs_eqn.shape[0]):
            h_ub_list.append(0.0)
            h_lb_list.append(-inf_val)
            slacked_list.append(0.0)
            h_rownames_list.append(input_constraint.name+'_r{:}_{:}'.format(idx, input_constraint.row_names[idx]))

    for sc_i, input_state_constraint in enumerate(mpc_env.input_state_constraints):
        cs_eqn = input_state_constraint.sym_cs_func(input=cs.vertcat(mpc_env.system_model.x_sym, mpc_env.system_model.u_sym))['con']
        h_expr_list.append(cs_eqn)
        slack_val = 1.0
        print('[OPT] input state constraint: {:}, {:}'.format(input_state_constraint.name, 'slacked' if slack_val != 0.0 else 'not slacked'))
        for idx in range(cs_eqn.shape[0]):
            h_ub_list.append(0.0)
            h_lb_list.append(-inf_val)
            slacked_list.append(slack_val)
            h_rownames_list.append(input_state_constraint.name+'_r{:}_{:}'.format(idx, input_state_constraint.row_names[idx]))


    kkt_const_idx = len(h_expr_list)
    if mpc_env.hum_model == 'orca_casadi_kkt' and mpc_env.orca_kkt_horiz > 0:
        K_orca = mpc_env.orca_kkt_horiz
        assert K_orca == mpc_env.horiz
        for _, con in enumerate(mpc_env.hum_orca_kkt_ineq_consts):
            # if mpc_env.human_pred_MID:
            #     cs_eqn = con.sym_cs_func(XU=cs.vertcat(mpc_env.system_model.x_sym, mpc_env.system_model.u_sym),
            #                              MID_samples_t_all_hums_stacked=mpc_env.MID_samples_t_all_hums_stacked,
            #                              MID_samples_tp1_all_hums_stacked=mpc_env.MID_samples_tp1_all_hums_stacked)['const']
            # else:
            if mpc_env.outdoor_robot_setting:
                cs_eqn = con.sym_cs_func(input=cs.vertcat(mpc_env.system_model.x_sym, mpc_env.system_model.u_sym, mpc_env.stat_obs_params_vecced))['const']
            else:
                cs_eqn = con.sym_cs_func(input=cs.vertcat(mpc_env.system_model.x_sym, mpc_env.system_model.u_sym))['const']
            h_expr_list.append(cs_eqn)
            slack_val  = 0.0
            print('[OPT] orca kkt ineq constraint: {:}, {:}'.format(con.name, 'slacked' if slack_val != 0.0 else 'not slacked'))
            for idx in range(cs_eqn.shape[0]):
                h_ub_list.append(0.0)
                h_lb_list.append(-inf_val)
                slacked_list.append(slack_val)
                h_rownames_list.append(con.name+'_r{:}_{:}'.format(idx, con.row_names[idx]))


        for _, con in enumerate(mpc_env.hum_orca_kkt_eq_consts):
            # if mpc_env.human_pred_MID:
            #     cs_eqn = con.sym_cs_func(XU=cs.vertcat(mpc_env.system_model.x_sym, mpc_env.system_model.u_sym),
            #                              MID_samples_t_all_hums_stacked=mpc_env.MID_samples_t_all_hums_stacked,
            #                              MID_samples_tp1_all_hums_stacked=mpc_env.MID_samples_tp1_all_hums_stacked)['const']
            # else:
            if mpc_env.outdoor_robot_setting:
                cs_eqn = con.sym_cs_func(input=cs.vertcat(mpc_env.system_model.x_sym, mpc_env.system_model.u_sym, mpc_env.stat_obs_params_vecced))['const']
            else:
                cs_eqn = con.sym_cs_func(input=cs.vertcat(mpc_env.system_model.x_sym, mpc_env.system_model.u_sym))['const']
            h_expr_list.append(cs_eqn)
            slack_val  = 0.0
            print('[OPT] orca kkt eq constraint: {:}, {:}'.format(con.name, 'slacked' if slack_val != 0.0 else 'not slacked'))
            for idx in range(cs_eqn.shape[0]):
                # # Try the Zac Manchester insight of adding a rho to the complementary slackness equality conditions. But everything just breaks...
                # if 'comp_slack' in con.row_names[idx]:
                #     h_ub_list.append(1e-7)
                #     h_lb_list.append(1e-7)
                # else:
                #     h_ub_list.append(0.0)
                #     h_lb_list.append(0.0)
                h_ub_list.append(0.0)
                h_lb_list.append(0.0)
                slacked_list.append(slack_val)
                h_rownames_list.append(con.name+'_r{:}_{:}'.format(idx, con.row_names[idx]))

        if mpc_env.human_pred_MID:
            cs_eqn = mpc_env.hums_close_to_preds_consts(XU=cs.vertcat(mpc_env.system_model.x_sym, mpc_env.system_model.u_sym), MID_samples_allhums_t_sym=cs.horzcat(*tuple(mpc_env.MID_samples_t_all_hums)))['const']
            slack_val  = 1.0
            print('[OPT] orca close to preds const {:}'.format('slacked' if slack_val != 0.0 else 'not slacked'))
            for idx in range(cs_eqn.shape[0]):
                h_expr_list.append(cs_eqn[idx])
                h_ub_list.append(0.0)
                h_lb_list.append(-inf_val)
                slacked_list.append(1.0)
                h_rownames_list.append('hums_close_to_preds_hum{:}'.format(idx))


    # Old way of slacking all equally
    # # set constraint values
    # model.con_h_expr = cs.vertcat(*h_expr_list)
    # model.con_h_expr_e = cs.vertcat(*h_e_expr_list)
    # ocp.model.con_h_expr = cs.vertcat(*h_expr_list)
    # ocp.model.con_h_expr_e = cs.vertcat(*h_e_expr_list)
    # ocp.constraints.lh = np.array(h_lb_list)
    # ocp.constraints.uh = np.array(h_ub_list)
    # ocp.constraints.lh_e = np.array(h_e_lb_list)
    # ocp.constraints.uh_e = np.array(h_e_ub_list)

    # # set which constraints are slacked (all of them)
    # # slack all of them equally
    # Jsh = np.eye(model.con_h_expr.shape[0])
    # Jsh_e = np.eye(model.con_h_expr_e.shape[0])

    # ocp.constraints.Jsh = Jsh
    # ocp.constraints.Jsh_e = Jsh_e
    # # set constraint slack penalties
    # L2_pen = 1e4
    # L1_pen = 10

    # ocp.cost.Zl = L2_pen * np.ones((model.con_h_expr.shape[0],))
    # ocp.cost.Zu = L2_pen * np.ones((model.con_h_expr.shape[0],))
    # ocp.cost.zl = L1_pen * np.ones((model.con_h_expr.shape[0],))
    # ocp.cost.zu = L1_pen * np.ones((model.con_h_expr.shape[0],))
    # ocp.cost.Zl_e = L2_pen * np.ones((model.con_h_expr_e.shape[0],))
    # ocp.cost.Zu_e = L2_pen * np.ones((model.con_h_expr_e.shape[0],))
    # ocp.cost.zl_e = L1_pen * np.ones((model.con_h_expr_e.shape[0],))
    # ocp.cost.zu_e = L1_pen * np.ones((model.con_h_expr_e.shape[0],))

    # Now sort expressions and bounds based on if they are slacked or not
    h_expr_list_SLACKED = []
    h_rowname_list_SLACKED = []
    h_ub_list_SLACKED = []
    h_lb_list_SLACKED = []
    h_expr_list_NOTSLACKED = []
    h_rowname_list_NOTSLACKED = []
    h_ub_list_NOTSLACKED = []
    h_lb_list_NOTSLACKED = []

    h_e_expr_list_SLACKED = []
    h_e_rowname_list_SLACKED = []
    h_e_ub_list_SLACKED = []
    h_e_lb_list_SLACKED = []
    h_e_expr_list_NOTSLACKED = []
    h_e_rowname_list_NOTSLACKED = []
    h_e_ub_list_NOTSLACKED = []
    h_e_lb_list_NOTSLACKED = []

    # the stage constraints
    b_idx = 0
    for idx in range(len(h_expr_list)):
        next_b_idx = b_idx + h_expr_list[idx].shape[0]
        if slacked_list[b_idx] == 0.0:
            h_expr_list_NOTSLACKED.append(h_expr_list[idx])
            h_ub_list_NOTSLACKED.extend(h_ub_list[b_idx:next_b_idx])
            h_lb_list_NOTSLACKED.extend(h_lb_list[b_idx:next_b_idx])
            h_rowname_list_NOTSLACKED.extend([h_rownames_list[c_idx]+'_HARD' for c_idx in range(b_idx,next_b_idx)])
        else:
            h_expr_list_SLACKED.append(h_expr_list[idx])
            h_ub_list_SLACKED.extend(h_ub_list[b_idx:next_b_idx])
            h_lb_list_SLACKED.extend(h_lb_list[b_idx:next_b_idx])
            h_rowname_list_SLACKED.extend([h_rownames_list[c_idx]+'_SLACKED' for c_idx in range(b_idx,next_b_idx)])
        b_idx = next_b_idx
    # the terminal constraints
    b_idx = 0
    for idx in range(len(h_e_expr_list)):
        next_b_idx = b_idx + h_e_expr_list[idx].shape[0]
        if slacked_e_list[idx] == 0.0:
            h_e_expr_list_NOTSLACKED.append(h_e_expr_list[idx])
            h_e_ub_list_NOTSLACKED.extend(h_e_ub_list[b_idx:next_b_idx])
            h_e_lb_list_NOTSLACKED.extend(h_e_lb_list[b_idx:next_b_idx])
            h_e_rowname_list_NOTSLACKED.extend([h_e_rownames_list[c_idx]+'_HARD' for c_idx in range(b_idx,next_b_idx)])
        else:
            h_e_expr_list_SLACKED.append(h_e_expr_list[idx])
            h_e_ub_list_SLACKED.extend(h_e_ub_list[b_idx:next_b_idx])
            h_e_lb_list_SLACKED.extend(h_e_lb_list[b_idx:next_b_idx])
            h_e_rowname_list_SLACKED.extend([h_e_rownames_list[c_idx]+'_SLACKED' for c_idx in range(b_idx,next_b_idx)])
        b_idx = next_b_idx

    # If the problem is not using hum_model orca_casadi_kkt, slack all constraints:
    if mpc_env.hum_model != 'orca_casadi_kkt':
        h_expr_list_SLACKED = h_expr_list_SLACKED + h_expr_list_NOTSLACKED
        h_expr_list_NOTSLACKED = []
        h_rowname_list_SLACKED = h_rowname_list_SLACKED + h_rowname_list_NOTSLACKED
        h_rowname_list_NOTSLACKED = []
        h_ub_list_SLACKED = h_ub_list_SLACKED + h_ub_list_NOTSLACKED
        h_ub_list_NOTSLACKED = []
        h_lb_list_SLACKED = h_lb_list_SLACKED + h_lb_list_NOTSLACKED
        h_lb_list_NOTSLACKED = []
        h_e_expr_list_SLACKED = h_e_expr_list_SLACKED + h_e_expr_list_NOTSLACKED
        h_e_expr_list_NOTSLACKED = []
        h_e_rowname_list_SLACKED = h_e_rowname_list_SLACKED + h_e_rowname_list_NOTSLACKED
        h_e_rowname_list_NOTSLACKED = []
        h_e_ub_list_SLACKED = h_e_ub_list_SLACKED + h_e_ub_list_NOTSLACKED
        h_e_ub_list_NOTSLACKED = []
        h_e_lb_list_SLACKED = h_e_lb_list_SLACKED + h_e_lb_list_NOTSLACKED
        h_e_lb_list_NOTSLACKED = []

    # set constraint values
    # Make debug functions of lists
    model.con_h_expr = cs.vertcat(*tuple(h_expr_list_SLACKED + h_expr_list_NOTSLACKED))
    model.con_h_expr_e = cs.vertcat(*tuple(h_e_expr_list_SLACKED + h_e_expr_list_NOTSLACKED))
    ocp.model.con_h_expr = cs.vertcat(*tuple(h_expr_list_SLACKED + h_expr_list_NOTSLACKED))
    ocp.model.con_h_expr_e = cs.vertcat(*tuple(h_e_expr_list_SLACKED + h_e_expr_list_NOTSLACKED))
    ocp.constraints.lh = np.array(h_lb_list_SLACKED + h_lb_list_NOTSLACKED)
    ocp.constraints.uh = np.array(h_ub_list_SLACKED + h_ub_list_NOTSLACKED)
    ocp.constraints.lh_e = np.array(h_e_lb_list_SLACKED + h_e_lb_list_NOTSLACKED)
    ocp.constraints.uh_e = np.array(h_e_ub_list_SLACKED + h_e_ub_list_NOTSLACKED)

    # Print out the names of the constraints list
    print('[OPT] Stage constraints:')
    all_stage_names = h_rowname_list_SLACKED + h_rowname_list_NOTSLACKED
    for idx, name in enumerate(all_stage_names):
        print('[OPT] {:}\t {:}'.format(idx, name))
    mpc_env.all_state_names = all_stage_names

    print('[OPT] Terminal constraints:')
    all_term_names = h_e_rowname_list_SLACKED + h_e_rowname_list_NOTSLACKED
    for idx, name in enumerate(all_term_names):
        print('[OPT] {:}\t {:}'.format(idx, name))
    mpc_env.all_term_names = all_term_names

    # set which constraints are slacked (all of them)
    # slack all of them equally
    Jsh = np.eye(len(h_expr_list_SLACKED))
    Jsh_e = np.eye(len(h_e_expr_list_SLACKED))

    ocp.constraints.Jsh = Jsh
    ocp.constraints.Jsh_e = Jsh_e
    # set constraint slack penalties
    L2_pen = 1e4
    L1_pen = 10

    ocp.cost.Zl = L2_pen * np.ones((len(h_expr_list_SLACKED),))
    ocp.cost.Zu = L2_pen * np.ones((len(h_expr_list_SLACKED),))
    ocp.cost.zl = L1_pen * np.ones((len(h_expr_list_SLACKED),))
    ocp.cost.zu = L1_pen * np.ones((len(h_expr_list_SLACKED),))
    ocp.cost.Zl_e = L2_pen * np.ones((len(h_e_expr_list_SLACKED),))
    ocp.cost.Zu_e = L2_pen * np.ones((len(h_e_expr_list_SLACKED),))
    ocp.cost.zl_e = L1_pen * np.ones((len(h_e_expr_list_SLACKED),))
    ocp.cost.zu_e = L1_pen * np.ones((len(h_e_expr_list_SLACKED),))






    # -------------------------------------------------------------------------------
    # set options
    # if mpc_env.hum_model == 'orca_casadi_kkt' and mpc_env.human_pred_MID or mpc_env.hum_model == 'cvmm':
    if mpc_env.hum_model == 'orca_casadi_kkt' or mpc_env.hum_model == 'cvmm':
        # -------------------------------------------------------------------------------
        # set options
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM' # 'PARTIAL_CONDENSING_HPIPM', 'FULL_CONDENSING_QPOASES', 'FULL_CONDENSING_HPIPM', 'PARTIAL_CONDENSING_QPDUNES', 'PARTIAL_CONDENSING_OSQP', 'FULL_CONDENSING_DAQP').
        # Default: 'PARTIAL_CONDENSING_HPIPM'

        ocp.solver_options.hessian_approx = 'EXACT' # 'GAUSS_NEWTON', 'EXACT'
        ocp.solver_options.regularize_method = 'PROJECT' # 'NO_REGULARIZE', 'MIRROR', 'PROJECT', 'PROJECT_REDUC_HESS', 'CONVEXIFY'
        ocp.solver_options.integrator_type = 'DISCRETE'
        ocp.solver_options.print_level = 0

        # set prediction horizon
        ocp.solver_options.tf = Tf
        # ocp.solver_options.nlp_solver_max_iter = 75
        # ocp.solver_options.nlp_solver_max_iter = 20 # WORKS FOR 2 HUMANS
        # ocp.solver_options.nlp_solver_max_iter = 12 # TOO SLOW FOR 3 HUMANS MID SAMPLES
        if mpc_env.isSim:
            ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
            ocp.solver_options.nlp_solver_max_iter = 50
            ocp.solver_options.qp_solver_iter_max = 15
            # ocp.solver_options.ext_fun_compile_flags = "-O0"
            # ocp.solver_options.ext_fun_compile_flags = "-O3 -mfma -ffast-math -mavx" # instead of -O2 which is the default, do more optimization (O3 is max)
            ocp.solver_options.ext_fun_compile_flags = "-O3 -ffast-math" # instead of -O2 which is the default, do more optimization (O3 is max)
        else:
            ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
            if mpc_env.hum_model == 'orca_casadi_kkt':
                if mpc_env.human_pred_MID:
                    if mpc_env.num_hums == 1:
                        ocp.solver_options.nlp_solver_max_iter = 10#15
                        ocp.solver_options.qp_solver_iter_max = 10 #20      # QP solver max iter # default: 50 WORKS
                    elif mpc_env.num_hums == 2:
                        ocp.solver_options.nlp_solver_max_iter = 8 #10
                        ocp.solver_options.qp_solver_iter_max = 8 #17       # QP solver max iter # default: 50 WORKS
                    elif mpc_env.num_hums == 3:
                        ocp.solver_options.nlp_solver_max_iter = 4
                        ocp.solver_options.qp_solver_iter_max = 5        # QP solver max iter # default: 50 WORKS
                    elif mpc_env.num_hums == 4:
                        ocp.solver_options.nlp_solver_max_iter = 3
                        ocp.solver_options.qp_solver_iter_max = 4        # QP solver max iter # default: 50 WORKS
                    else:
                        ocp.solver_options.nlp_solver_max_iter = 3
                        ocp.solver_options.qp_solver_iter_max = 4        # QP solver max iter # default: 50 WORKS
                else:
                    # To run at 10 Hz
                    if mpc_env.num_hums == 1:
                        ocp.solver_options.nlp_solver_max_iter = 10
                        ocp.solver_options.qp_solver_iter_max = 10
                    elif mpc_env.num_hums == 2:
                        ocp.solver_options.nlp_solver_max_iter = 8
                        ocp.solver_options.qp_solver_iter_max = 10
                    elif mpc_env.num_hums == 3:
                        ocp.solver_options.nlp_solver_max_iter = 5
                        ocp.solver_options.qp_solver_iter_max = 5
                    elif mpc_env.num_hums == 4:
                        ocp.solver_options.nlp_solver_max_iter = 3
                        ocp.solver_options.qp_solver_iter_max = 2
            else:

                ocp.solver_options.nlp_solver_max_iter = 20
                ocp.solver_options.qp_solver_iter_max = 20
                # ocp.solver_options.qp_solver_iter_max = 4*15        # QP solver max iter # default: 50 WORKS
                # ocp.solver_options.qp_solver_tol_stat = 1e-6                    # NLP solver stationarity tolerance
                # ocp.solver_options.qp_solver_tol_eq   = 1e-6                     # NLP solver equality tolerance
                # ocp.solver_options.qp_solver_tol_ineq = 1e-6                     # NLP solver inequality
                # ocp.solver_options.qp_solver_tol_comp = 1e-6                    # NLP solver complementarity

            # ocp.solver_options.ext_fun_compile_flags = "-O3 -mfma -ffast-math -mavx" # instead of -O2 which is the default, do more optimization (O3 is max)
            ocp.solver_options.ext_fun_compile_flags = "-O3 -ffast-math" # instead of -O2 which is the default, do more optimization (O3 is max)
            # ocp.solver_options.ext_fun_compile_flags = "-O0" # instead of -O2 which is the default and does some optimization
            # ocp.solver_options.ext_fun_compile_flags = "-O0 -flto=8" # not sure what flto does

        ocp.solver_options.nlp_solver_tol_stat = 1e-3                    # NLP solver stationarity tolerance
        ocp.solver_options.nlp_solver_tol_eq   = 1e-3                     # NLP solver equality tolerance
        ocp.solver_options.nlp_solver_tol_ineq = 1e-3                     # NLP solver inequality
        ocp.solver_options.nlp_solver_tol_comp = 1e-3                    # NLP solver complementarity


        # ocp.solver_options.qp_solver_tol_stat = None      # QP solver stationarity tolerance # default: None
        # ocp.solver_options.qp_solver_tol_eq   = None      # QP solver equality tolerance # default: None
        # ocp.solver_options.qp_solver_tol_ineq = None      # QP solver inequality # default: None
        # ocp.solver_options.qp_solver_tol_comp = None      # QP solver complementarity # default: None
        # ocp.solver_options.qp_solver_iter_max = 25        # QP solver max iter WORKS FOR 1-3 HUMANS.
        # ocp.solver_options.qp_solver_iter_max = 20        # QP solver max iter # default: 50 TOO SLOW WITH 3 HUMANS 10 MID SAMPLES

        # ocp.solver_options.qp_solver_cond_N = None        # QP solver: new horizon after partial condensing # default: None
        ocp.solver_options.qp_solver_warm_start = 1       # default: 0
        # ocp.solver_options.qp_solver_cond_ric_alg = 1     # default: 1
        # ocp.solver_options.qp_solver_ric_alg = 1          # default: 1
    else:
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES' # 'PARTIAL_CONDENSING_HPIPM', 'FULL_CONDENSING_QPOASES', 'FULL_CONDENSING_HPIPM', 'PARTIAL_CONDENSING_QPDUNES', 'PARTIAL_CONDENSING_OSQP', 'FULL_CONDENSING_DAQP').
        # Default: 'PARTIAL_CONDENSING_HPIPM'

        ocp.solver_options.hessian_approx = 'EXACT' # 'GAUSS_NEWTON', 'EXACT'
        ocp.solver_options.regularize_method = 'PROJECT' # 'NO_REGULARIZE', 'MIRROR', 'PROJECT', 'PROJECT_REDUC_HESS', 'CONVEXIFY'
        ocp.solver_options.integrator_type = 'DISCRETE'
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
        # set prediction horizon
        ocp.solver_options.tf = Tf
        # ocp.solver_options.nlp_solver_max_iter = 75
        # ocp.solver_options.nlp_solver_max_iter = 20 # WORKS FOR 2 HUMANS
        # ocp.solver_options.nlp_solver_max_iter = 12 # TOO SLOW FOR 3 HUMANS MID SAMPLES
        ocp.solver_options.nlp_solver_max_iter = 10 if not mpc_env.isSim else 100
        ocp.solver_options.nlp_solver_tol_stat = 1e-2                    # NLP solver stationarity tolerance
        ocp.solver_options.nlp_solver_tol_eq   = 1e-2                     # NLP solver equality tolerance
        ocp.solver_options.nlp_solver_tol_ineq = 1e-2                     # NLP solver inequality
        ocp.solver_options.nlp_solver_tol_comp = 1e-2                    # NLP solver complementarity

        # ocp.solver_options.ext_fun_compile_flags = "-O0" # instead of -O2 which is the default and does some optimization
        # ocp.solver_options.ext_fun_compile_flags = "-O0 -flto=8" # not sure what flto does

        # ocp.solver_options.ext_fun_compile_flags = "-O3 -ffast-math -mavx" # instead of -O2 which is the default, do more optimization (O3 is max)



        # ocp.solver_options.qp_solver_tol_stat = None      # QP solver stationarity tolerance # default: None
        # ocp.solver_options.qp_solver_tol_eq   = None      # QP solver equality tolerance # default: None
        # ocp.solver_options.qp_solver_tol_ineq = None      # QP solver inequality # default: None
        # ocp.solver_options.qp_solver_tol_comp = None      # QP solver complementarity # default: None
        # ocp.solver_options.qp_solver_iter_max = 25        # QP solver max iter WORKS FOR 1-3 HUMANS.
        # ocp.solver_options.qp_solver_iter_max = 20        # QP solver max iter # default: 50 TOO SLOW WITH 3 HUMANS 10 MID SAMPLES
        ocp.solver_options.qp_solver_iter_max = 15        # QP solver max iter # default: 50 WORKS
        # ocp.solver_options.qp_solver_cond_N = None        # QP solver: new horizon after partial condensing # default: None
        ocp.solver_options.qp_solver_warm_start = 1       # default: 0
        # ocp.solver_options.qp_solver_cond_ric_alg = 1     # default: 1
        # ocp.solver_options.qp_solver_ric_alg = 1          # default: 1

    # -------------------------------------------------------------------------------
    # check if json file is existing file and if the folder is an existing folder


    # solver = AcadosOcpSolver(ocp, json_file = 'acados_mpc_cvmm_ocp.json')

    solver = AcadosOcpSolver(ocp, json_file=json_file, generate=regen, build=regen)
    # solver.build(code_export_dir=code_export_dir, with_cython=True)
    # solver.build(code_export_dir=code_export_dir)

    # simX = np.ndarray((N+1, nx))
    # simU = np.ndarray((N, nu))


    return solver, ocp