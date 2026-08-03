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

class Section2Scene(TeachingScene):
    def construct(self):
        # Setup
        title_text = "The Pattern: A Mysterious Number Emerges"
        lecture_lines = [
            "Let the large block move toward the wall.",
            "If masses are equal, we count three collisions.",
            "Increase the large mass by factors of one hundred.",
            "The collision counts remarkably reveal digits of pi.",
            "Why does this physical system calculate pi?"
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_B = "#00FFFF" 
        COLOR_A = "#FF8800" 
        COLOR_WALL = "#888888"
        COLOR_PI = "#FFFF00"
        COLOR_QUESTION = "#00FF00"

        # Assets
        # Issue 25: Integrate SVG assets
        wall_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg", color=COLOR_WALL)
        self.place_at_grid(wall_svg, "D1", scale_factor=2.0)
        wall_x = wall_svg.get_right()[0]
        
        floor = Line(self.grid["F1"] + LEFT*0.5, self.grid["F6"] + RIGHT*0.5, color=COLOR_WALL, stroke_width=4)
        
        # Scoreboard
        # Issue 25: Scoreboard asset
        score_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scoreboard.svg", height=0.4)
        score_label = Text("Collisions:", font_size=20, color=WHITE)
        self.score_num = Integer(0, font_size=28, color=WHITE)
        self.scoreboard = VGroup(score_icon, score_label, self.score_num).arrange(RIGHT, buff=0.2)
        # Issue 37: Fix scoreboard placement
        self.place_in_area(self.scoreboard, "A5", "A6", scale_factor=0.8)
        
        # Blocks
        # Issue 37: Fix block A placement at F3
        self.block_a = Square(side_length=0.4, fill_opacity=1, color=COLOR_A).set_stroke(WHITE, 1)
        self.place_at_grid(self.block_a, "F3")
        self.block_a.shift(UP*0.2) # Align with floor top

        # Issue 37: Fix block B placement at F6
        self.block_b = Square(side_length=0.8, fill_opacity=1, color=COLOR_B).set_stroke(WHITE, 1)
        self.place_at_grid(self.block_b, "F6")
        self.block_b.shift(UP*0.4) # Align with floor top

        label_a = Text("m", font_size=18, color=WHITE)
        label_b = Text("M", font_size=22, color=WHITE)
        label_a.next_to(self.block_a, UP, buff=0.1)
        label_b.next_to(self.block_b, UP, buff=0.1)

        # Add initial objects
        self.add(wall_svg, floor, self.scoreboard, self.block_a, self.block_b, label_a, label_b)

        # === Animation for Lecture Line 1 ===
        # "Let the large block move toward the wall."
        self.lecture[0].set_color(WHITE)
        
        # Updaters for labels
        label_a.add_updater(lambda m: m.next_to(self.block_a, UP, buff=0.1))
        label_b.add_updater(lambda m: m.next_to(self.block_b, UP, buff=0.1))

        # Initial approach of Block B
        self.play(self.block_b.animate.shift(LEFT * 1.0), run_time=1.5)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # "If masses are equal, we count three collisions."
        self.lecture[1].set_color(COLOR_B)
        
        # M=1 physics sequence (Simplified)
        # 1. B hits A
        self.play(self.block_b.animate.next_to(self.block_a, RIGHT, buff=0), run_time=0.4)
        self.score_num.set_value(1)
        # 2. A hits wall
        dist_to_wall = self.block_a.get_left()[0] - wall_x
        self.play(self.block_a.animate.shift(LEFT * dist_to_wall), run_time=0.4)
        self.score_num.set_value(2)
        # 3. A hits B
        self.play(self.block_a.animate.next_to(self.block_b, LEFT, buff=0), run_time=0.4)
        self.score_num.set_value(3)
        # Separation
        self.play(self.block_b.animate.shift(RIGHT * 2.0), self.block_a.animate.shift(RIGHT * 0.5), run_time=0.8)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # "Increase the large mass by factors of one hundred."
        self.lecture[2].set_color(COLOR_PI)
        
        # Reset positions for the M=100 demonstration
        self.play(
            self.score_num.animate.set_value(0),
            self.block_a.animate.move_to(self.grid["F3"] + UP*0.2),
            self.block_b.animate.move_to(self.grid["F6"] + UP*0.4)
        )
        mass_text = Text("M = 100 m", font_size=20, color=COLOR_PI)
        self.place_at_grid(mass_text, "E6")
        mass_text.add_updater(lambda m: m.next_to(self.block_b, DOWN, buff=0.1))
        self.play(Write(mass_text))

        # === Animation for Lecture Line 4 ===
        # "The collision counts remarkably reveal digits of pi."
        self.lecture[3].set_color(COLOR_PI)

        # M=100 Simulation Loop
        m1 = 1
        m2 = 100
        v1 = 0
        v2 = -1.5 # Initial velocity of Block B
        hits = 0
        
        # Use simple simulation steps to avoid heavy always_redraw
        while hits < 31:
            if v1 < 0: # A moves left towards wall
                dist = self.block_a.get_left()[0] - wall_x
                dt = dist / abs(v1)
                anim_time = 0.05
                self.play(
                    self.block_a.animate.shift(LEFT * dist),
                    self.block_b.animate.shift(RIGHT * (v2 * dt)),
                    run_time=anim_time, rate_func=linear
                )
                v1 = -v1
                hits += 1
                self.score_num.set_value(hits)
            else: # A moves right or B moves left (collision between blocks)
                dist = self.block_b.get_left()[0] - self.block_a.get_right()[0]
                rel_v = v1 - v2
                if rel_v <= 1e-6: break 
                
                dt = dist / rel_v
                anim_time = 0.05
                self.play(
                    self.block_a.animate.shift(RIGHT * (v1 * dt)),
                    self.block_b.animate.shift(RIGHT * (v2 * dt)),
                    run_time=anim_time, rate_func=linear
                )
                
                # Elastic collision
                v1_n = (m1 - m2)/(m1 + m2) * v1 + 2*m2/(m1 + m2) * v2
                v2_n = 2*m1/(m1 + m2) * v1 + (m2 - m1)/(m1 + m2) * v2
                v1, v2 = v1_n, v2_n
                hits += 1
                self.score_num.set_value(hits)

        # Separation
        self.play(
            self.block_a.animate.shift(RIGHT * 1.0),
            self.block_b.animate.shift(RIGHT * 1.5),
            run_time=1
        )

        # === Animation for Lecture Line 5 ===
        # "Why does this physical system calculate pi?"
        self.lecture[4].set_color(COLOR_QUESTION)
        
        # Highlight '31'
        self.play(self.score_num.animate.set_color(COLOR_PI).scale(1.5))
        self.play(Flash(self.score_num, color=COLOR_PI))
        self.wait(3)
