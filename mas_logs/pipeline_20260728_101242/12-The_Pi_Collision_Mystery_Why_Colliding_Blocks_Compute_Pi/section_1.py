from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section1Scene(TeachingScene):
    def construct(self):
        # 1. Setup Layout
        lecture_lines = [
            "Two blocks and a wall on a frictionless surface.",
            "Large block mass is a power of 100.",
            "Total collisions reveal the digits of Pi."
        ]
        self.setup_layout("The Impossible Experiment", lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Wall [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg]
        wall = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg")
        wall.set_color(WHITE)
        # Fix for Issue 21: Position wall at C3 to avoid overlapping lecture notes
        self.place_at_grid(wall, 'C3', scale_factor=0.8)
        wall.stretch_to_fit_height(2.5)
        
        # Physics boundaries
        wall_x = wall.get_right()[0]
        floor_y = self.grid["D3"][1] - 0.2
        
        # Floor
        floor = Line(self.grid["D3"] + LEFT*0.5, self.grid["D6"] + RIGHT*0.5, color="#333333", stroke_width=4)
        floor.set_y(floor_y)
        
        self.play(Create(floor), FadeIn(wall))
        
        # Blocks [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg]
        small_block = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg")
        small_block.set_color("#0000FF")
        small_block.height = 0.4
        
        large_block = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg")
        large_block.set_color("#00FF00")
        large_block.height = 0.8
        
        # Starting simulation coordinates
        curr_x1 = wall_x + 0.6
        curr_x2 = wall_x + 2.5
        
        small_block.move_to([curr_x1, floor_y + 0.2, 0])
        large_block.move_to([curr_x2, floor_y + 0.4, 0])
        
        m_label = MathTex("m=1", font_size=20, color="#0000FF")
        M_label = MathTex("M=100", font_size=20, color="#00FF00")
        
        # Persistent Labels (Constraint 10)
        m_label.add_updater(lambda m: m.next_to(small_block, UP, buff=0.1))
        M_label.add_updater(lambda m: m.next_to(large_block, UP, buff=0.1))
        
        self.play(FadeIn(small_block), FadeIn(large_block), FadeIn(m_label), FadeIn(M_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        self.play(Indicate(M_label, color=YELLOW, scale_factor=1.2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Collision Counter setup
        # Fix for Issue 22: Move counter to B5 for better layout balance
        collision_count_tracker = ValueTracker(0)
        counter_label = Text("Collisions:", font_size=24, color="#FFFF00")
        counter_val = Integer(0, color="#FFFF00")
        counter_val.add_updater(lambda d: d.set_value(int(collision_count_tracker.get_value())))
        counter_group = VGroup(counter_label, counter_val).arrange(RIGHT, buff=0.2)
        self.place_at_grid(counter_group, 'B5', scale_factor=0.9)
        self.add(counter_group)
        
        # Simulation: M=100 results in 31 collisions
        m1 = 1.0
        m2 = 100.0
        v1 = 0.0
        v2 = -2.0 # Initial velocity
        
        r1 = 0.2 # half-width
        r2 = 0.4 # half-width
        
        states = []
        c = 0
        
        # Calculate exactly 31 collisions
        while c < 31:
            t_wall = float('inf')
            if v1 < 0:
                t_wall = (wall_x + r1 - curr_x1) / v1
            
            t_blocks = float('inf')
            if v1 - v2 > 0:
                t_blocks = (curr_x2 - r2 - (curr_x1 + r1)) / (v1 - v2)
            
            if t_wall < t_blocks and t_wall > 0:
                dt_hit = t_wall
                curr_x1 += v1 * dt_hit
                curr_x2 += v2 * dt_hit
                v1 = -v1
                c += 1
                states.append({'x1': curr_x1, 'x2': curr_x2, 'c': c, 'dt': dt_hit})
            elif t_blocks < t_wall and t_blocks > 0:
                dt_hit = t_blocks
                curr_x1 += v1 * dt_hit
                curr_x2 += v2 * dt_hit
                v1_new = ((m1 - m2) / (m1 + m2)) * v1 + ((2 * m2) / (m1 + m2)) * v2
                v2_new = ((2 * m1) / (m1 + m2)) * v1 + ((m2 - m1) / (m1 + m2)) * v2
                v1, v2 = v1_new, v2_new
                c += 1
                states.append({'x1': curr_x1, 'x2': curr_x2, 'c': c, 'dt': dt_hit})
            else:
                break
        
        # Execute calculated animations within performance budget
        for i, state in enumerate(states):
            run_time = 0.06 if i < 10 else 0.03
            if i > 20: run_time = 0.015
                
            self.play(
                small_block.animate(run_time=run_time, rate_func=linear).set_x(state['x1']),
                large_block.animate(run_time=run_time, rate_func=linear).set_x(state['x2']),
                collision_count_tracker.animate(run_time=run_time).set_value(state['c']),
            )
            
        # Final movement: Both blocks move away to the right
        self.play(
            small_block.animate.shift(RIGHT * 5),
            large_block.animate.shift(RIGHT * 3),
            run_time=1.5
        )
        self.wait(2)
