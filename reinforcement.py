import numpy as np
import gym

# Q-learning parametrləri
env = gym.make('FrozenLake-v1')  # Mühiti seçirik
Q = np.zeros([env.observation_space.n, env.action_space.n])  # Q cədvəli
learning_rate = 0.8
discount_factor = 0.95
episodes = 1000

# Q-learning alqoritmasını tətbiq edirik
for i in range(episodes):
    state = env.reset()  # state burada tuple ola bilər
    done = False
    
    while not done:
        # state dəyişənini tam ədədə çevirmək
        state = state[0] if isinstance(state, tuple) else state  # Tuple-dan sadəcə ilk elementini alırıq
        
        # Hərəkət seçimi (epsilon-greedy)
        if np.random.uniform(0, 1) < 0.1:
            action = env.action_space.sample()  # Təsadüfi hərəkət
        else:
            action = np.argmax(Q[state, :])  # Ən yaxşı hərəkət
        
        # Mühitə hərəkət göndəririk və yeni vəziyyəti alırıq
        step_result = env.step(action)  # 4 dəyər qaytarır
        
        # Nə qaytarıldığını çap edək
        print("Step result:", step_result)  # Bu sətir ilə qaytarılan dəyərləri görə bilərik
        
        # Step nəticəsinin uzunluğuna görə dəyərləri ayırırıq
        if len(step_result) == 4:  # 4 dəyər qaytarılır
            next_state, reward, done, info = step_result
        elif len(step_result) == 3:  # 3 dəyər qaytarılır
            next_state, reward, done = step_result
            info = None  # `info` yoxdur
        
        # next_state-də də eyni yoxlamanı edirik
        if isinstance(next_state, tuple):
            next_state = next_state[0]  # Tuple-dan sadəcə ilk elementini alırıq
        
        # Q cədvəlini yeniləyirik
        Q[state, action] = Q[state, action] + learning_rate * (reward + discount_factor * np.max(Q[next_state, :]) - Q[state, action])
        
        state = next_state  # next_state-ni yeni state olaraq təyin edirik

# Təlim tamamlandıqdan sonra agentin öyrəndiyi Q cədvəlini çap edə bilərik
print(Q)
