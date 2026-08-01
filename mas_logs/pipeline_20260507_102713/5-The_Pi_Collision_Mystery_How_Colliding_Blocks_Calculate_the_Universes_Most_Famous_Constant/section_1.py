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
        # Setup the scene layout with title and lecture lines
        lecture_lines = [
            'Imagine two blocks on a frictionless floor.', 
            'They collide with a wall and each other.', 
            'For specific mass ratios, collisions reveal Pi.'
        ]
        self.setup_layout("The Hook: The Counting Machine", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # "Imagine two blocks on a frictionless floor."
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Floor (#FFFFFF) and wall (#FFFFFF) at the left
        # Floor sits at Row F, Wall starts at Column 1
        floor = Line(self.grid["F1"], self.grid["F6"], color="#FFFFFF")
        wall = Line(self.grid["A1"], self.grid["F1"], color="#FFFFFF")
        
        # Small red square (mass m) - #FF0000
        block_m = Square(side_length=0.4, fill_opacity=1, stroke_width=2, color="#FF0000")
        block_m.move_to(self.grid["F2"] + UP * 0.2)
        
        # Massive blue square (mass M) - #0000FF
        block_M = Square(side_length=0.8, fill_opacity=1, stroke_width=2, color="#0000FF")
        block_M.move_to(self.grid["F5"] + UP * 0.4)
        
        self.add(floor, wall, block_m, block_M)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "They collide with a wall and each other."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Collision Counter scoreboard above the blocks (#D3D3D3)
        counter_label = Text("Collisions:", font_size=24, color="#D3D3D3")
        count_num = Integer(0, font_size=24, color="#D3D3D3", mob_class=Text)
        counter_group = VGroup(counter_label, count_num).arrange(RIGHT, buff=0.2)
        
        # Fix: Using place_in_area for group and shifting to Row A (Issues 25 & 26)
        self.place_in_area(counter_group, 'A3', 'B4', scale_factor=1.2)
        self.add(counter_group)
        
        # Physics parameters: m=1, M=100 (yielding 31 collisions)
        m1 = 1
        m2 = 100
        v1 = 0.0
        v2 = -1.8  # Initial velocity of large block moving left
        
        # State variables for simulation
        x1 = block_m.get_center()[0]
        x2 = block_M.get_center()[0]
        s1, s2 = 0.4, 0.8
        wall_x = self.grid["F1"][0]
        collision_count = 0
        
        def physics_update(mob, dt):
            nonlocal v1, v2, x1, x2, collision_count
            
            # Sub-stepping for stable collision detection
            substeps = 40
            dt_sub = dt / substeps
            
            for _ in range(substeps):
                x1 += v1 * dt_sub
                x2 += v2 * dt_sub
                
                # Check collision with wall
                if x1 - s1/2 <= wall_x:
                    v1 = -v1
                    x1 = wall_x + s1/2
                    collision_count += 1
                
                # Check collision between blocks
                if x2 - s2/2 <= x1 + s1/2:
                    # Perfectly elastic collision formula
                    v1_new = ((m1 - m2) / (m1 + m2)) * v1 + (2 * m2 / (m1 + m2)) * v2
                    v2_new = (2 * m1 / (m1 + m2)) * v1 + ((m2 - m1) / (m1 + m2)) * v2
                    v1, v2 = v1_new, v2_new
                    
                    # Prevent penetration
                    x2 = x1 + (s1/2 + s2/2)
                    collision_count += 1
            
            # Update mobject positions and scoreboard
            block_m.set_x(x1)
            block_M.set_x(x2)
            count_num.set_value(collision_count)

        # Apply updater to trigger simulation
        block_m.add_updater(physics_update)
        
        # Run until the process is complete (31 collisions)
        self.wait(6)
        block_m.remove_updater(physics_update)
        
        # Final buffer to show the counter at 31
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # "For specific mass ratios, collisions reveal Pi."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        self.wait(2.5)
        
        # Cleanup colors
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
