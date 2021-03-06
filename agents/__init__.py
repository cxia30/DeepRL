from agents.registration import register_agent, make_agent

register_agent(name="A2C",
               entry_point="agents.a2c:A2CDiscrete",
               state_dimensions="single",
               action_space="discrete"
               )
register_agent(name="A2C",
               entry_point="agents.a2c:A2CContinuous",
               state_dimensions="single",
               action_space="continuous"
               )
register_agent(name="A3C",
               entry_point="agents.a3c:A3CDiscrete",
               state_dimensions="single",
               action_space="discrete"
               )
register_agent(name="A3C",
               entry_point="agents.a3c:A3CDiscreteCNN",
               state_dimensions="multi",
               action_space="discrete"
               )
register_agent(name="A3C",
               entry_point="agents.a3c:A3CDiscreteCNNRNN",
               state_dimensions="multi",
               action_space="discrete",
               RNN=True
               )
register_agent(name="A3C",
               entry_point="agents.a3c:A3CContinuous",
               state_dimensions="single",
               action_space="continuous"
               )
register_agent(name="AsyncKnowledgeTransfer",
               entry_point="agents.async_knowledge_transfer:AsyncKnowledgeTransfer",
               state_dimensions="single",
               action_space="discrete"
               )
register_agent(name="CEM",
               entry_point="agents.cem:CEM",
               state_dimensions="single",
               action_space="discrete"
               )
register_agent(name="CEM",
               entry_point="agents.cem:CEM",
               state_dimensions="single",
               action_space="continuous"
               )
register_agent(name="Karpathy",
               entry_point="agents.karpathy:Karpathy",
               state_dimensions="single",
               action_space="discrete"
               )
register_agent(name="KnowledgeTransfer",
               entry_point="agents.knowledge_transfer:KnowledgeTransfer",
               state_dimensions="single",
               action_space="discrete"
               )
register_agent(name="REINFORCE",
               entry_point="agents.reinforce:REINFORCEDiscrete",
               state_dimensions="single",
               action_space="discrete"
               )
register_agent(name="REINFORCE",
               entry_point="agents.reinforce:REINFORCEDiscreteRNN",
               state_dimensions="single",
               action_space="discrete",
               RNN=True
               )
register_agent(name="REINFORCE",
               entry_point="agents.reinforce:REINFORCEDiscreteCNN",
               state_dimensions="multi",
               action_space="discrete"
               )
register_agent(name="REINFORCE",
               entry_point="agents.reinforce:REINFORCEDiscreteCNNRNN",
               state_dimensions="multi",
               action_space="discrete",
               RNN=True
               )
register_agent(name="REINFORCE",
               entry_point="agents.reinforce:REINFORCEContinuous",
               state_dimensions="single",
               action_space="continuous"
               )
register_agent(name="SarsaFA",
               entry_point="agents.sarsa_fa:SarsaFA",
               state_dimensions="single",
               action_space="discrete"
               )
