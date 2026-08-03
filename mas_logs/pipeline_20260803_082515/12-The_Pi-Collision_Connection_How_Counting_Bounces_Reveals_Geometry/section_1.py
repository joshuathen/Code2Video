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
        # Setup layout with title and lecture lines
        self.setup_layout(
            "The Setup: A Strange Counting Game",
            [
                "Imagine two blocks on a frictionless surface.",
                "A wall sits to the far left.",
                "We count every bounce between blocks and the wall."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.play(self.lecture[0].animate.set_color(BLUE_B))
        
        # Floor line positioned at the base of row F
        floor_y = self.grid['F1'][1] - 0.5
        floor = Line(
            [self.grid['A1'][0] - 0.5, floor_y, 0], 
            [self.grid['F6'][0] + 0.5, floor_y, 0], 
            color=WHITE
        )
        
        # Wall asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg
        # Positioning at column 1, rows A to F (Asset: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg])
        wall = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg")
        self.place_in_area(wall, "A1", "F1", scale_factor=2.0)
        
        # Small block m (blue) placed at F2 per Issue 36
        block_m = Square(side_length=0.6, fill_opacity=0.8, color=BLUE_B)
        self.place_at_grid(block_m, 'F2')
        # Label m at E2
        m_label = MathTex("m", color=BLUE_B)
        self.place_at_grid(m_label, 'E2')
        
        # Large block M (yellow) placed at F5 per Issue 36
        block_M = Square(side_length=1.0, fill_opacity=0.8, color=YELLOW_B)
        self.place_at_grid(block_M, 'F5')
        # Label M at E5
        M_label = MathTex("M", color=YELLOW_B)
        self.place_at_grid(M_label, 'E5')
        
        self.play(Create(floor), FadeIn(wall))
        self.play(FadeIn(block_m), FadeIn(m_label), FadeIn(block_M), FadeIn(M_label))
        
        # Block M starts sliding to F3
        self.play(
            block_M.animate.move_to(self.grid['F3']), 
            M_label.animate.move_to(self.grid['E3']), 
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Update lecture line highlights
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(BLUE_B)
        )
        
        # Collision counter setup per Issue 36
        counter_val = ValueTracker(0)
        counter_label = Text("Collisions:", font_size=24, color=WHITE)
        self.place_in_area(counter_label, 'A3', 'A4', scale_factor=0.8)
        
        counter_num = DecimalNumber(0, num_decimal_places=0, color=WHITE)
        counter_num.add_updater(lambda d: d.set_value(counter_val.get_value()))
        self.place_at_grid(counter_num, 'A5')
        
        self.play(FadeIn(counter_label), FadeIn(counter_num))
        
        # First collision: M hits m
        # Move them towards each other (collision near F2/F3 border)
        collision_x = (self.grid['F2'][0] + self.grid['F3'][0]) / 2
        collision_pos_m = np.array([collision_x - 0.3, self.grid['F2'][1], 0])
        collision_pos_M = np.array([collision_x + 0.5, self.grid['F3'][1], 0])
        
        self.play(
            block_M.animate.move_to(collision_pos_M),
            M_label.animate.move_to(collision_pos_M + UP*1.0),
            block_m.animate.move_to(collision_pos_m),
            m_label.animate.move_to(collision_pos_m + UP*1.0),
            run_time=0.5
        )
        
        # Flash and increment
        self.play(
            counter_val.animate.set_value(1), 
            Flash([collision_x, self.grid['F2'][1], 0], color=WHITE)
        )
        
        # Block m moves towards wall (column 1)
        wall_pos_m = np.array([self.grid['F1'][0] + 0.3, self.grid['F1'][1], 0])
        self.play(
            block_m.animate.move_to(wall_pos_m), 
            m_label.animate.move_to(wall_pos_m + UP*1.0),
            run_time=0.8
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Update lecture line highlights
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW_B)
        )
        
        # Block m hits wall (at column 1 center x=0.5)
        # Asset: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg]
        wall_hit_x = self.grid['F1'][0] + 0.3
        self.play(
            block_m.animate.move_to([wall_hit_x, self.grid['F1'][1], 0]),
            m_label.animate.move_to([wall_hit_x, self.grid['F1'][1] + 1.0, 0]),
            run_time=0.2
        )
        self.play(
            counter_val.animate.set_value(2), 
            Flash([self.grid['F1'][0], self.grid['F1'][1], 0], color=WHITE)
        )
        
        # Block m bounces back and hits M again
        self.play(
            block_m.animate.move_to(collision_pos_m),
            m_label.animate.move_to(collision_pos_m + UP*1.0),
            run_time=0.5
        )
        
        self.play(
            counter_val.animate.set_value(3), 
            Flash([collision_x, self.grid['F2'][1], 0], color=WHITE)
        )
        
        # Final slow movement away to signify continuing bounces
        self.play(
            block_m.animate.move_to(self.grid['F2']),
            m_label.animate.move_to(self.grid['E2']),
            block_M.animate.move_to(self.grid['F4']),
            M_label.animate.move_to(self.grid['E4']),
            run_time=1
        )
        self.wait(2)
