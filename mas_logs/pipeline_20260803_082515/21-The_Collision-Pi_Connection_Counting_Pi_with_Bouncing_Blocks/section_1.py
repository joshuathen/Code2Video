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
        self.setup_layout(
            "The Setup: The Ant and the Elephant",
            [
                "Imagine two blocks on a frictionless floor.",
                "Small block sits between a wall and large block.",
                "We count every collision until the blocks separate."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Draw white #FFFFFF wall on left and green #00FF00 ground line.
        self.lecture[0].set_color(YELLOW)
        
        # Wall: A vertical line in the first column of the grid
        wall = Line(self.grid["A1"], self.grid["F1"], color=WHITE, stroke_width=6)
        # Floor: A horizontal line in the bottom row
        floor = Line(self.grid["F1"], self.grid["F6"], color=GREEN, stroke_width=4)
        
        self.play(Create(wall), Create(floor))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Place small blue #0000FF Block A and large red #FF0000 Block B.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Block A (Small) - placed at E2 (sitting on the floor)
        block_a = Square(side_length=0.6, color=BLUE, fill_opacity=1.0)
        self.place_at_grid(block_a, "E2")
        
        # Block B (Large) - placed at E5 (sitting on the floor)
        block_b = Square(side_length=1.2, color=RED, fill_opacity=1.0)
        self.place_at_grid(block_b, "E5")
        
        # Labels within 1 grid unit
        label_a = Text("A", font_size=20, color=WHITE)
        label_b = Text("B", font_size=20, color=WHITE)
        
        self.place_at_grid(label_a, "D2")
        self.place_at_grid(label_b, "D5")
        
        self.play(FadeIn(block_a), FadeIn(block_b), Write(label_a), Write(label_b))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Animate Block B sliding left at constant velocity towards Block A.
        # Flash Block A yellow #FFFF00 when it hits the wall or Block B.
        # Show 'Total Collisions: 0' text at top center in white #FFFFFF.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Persistent collision counter
        collision_counter_label = Text("Total Collisions: ", font_size=24, color=WHITE)
        collision_count = Integer(0, font_size=24, color=WHITE)
        counter_group = VGroup(collision_counter_label, collision_count).arrange(RIGHT, buff=0.1)
        # Fix for Issue 28: Use place_in_area for better centering and avoid label overlap
        self.place_in_area(counter_group, "A2", "A5", scale_factor=0.7)
        
        self.play(FadeIn(counter_group))
        
        # Define positions for a simple simulation sequence
        # Target for Block B to collide with A
        # A is at E2 (center x=1.5, half-width 0.3, right edge 1.8)
        # B is at E5 (center x=4.5, half-width 0.6, left edge 3.9)
        # Collision occurs when B's left edge (x - 0.6) = A's right edge (1.8)
        # So x = 2.4
        target_b_x = 2.4
        target_b_pos = np.array([target_b_x, self.grid["E5"][1], 0])
        
        # Slide B and its label
        self.play(
            block_b.animate.move_to(target_b_pos),
            label_b.animate.move_to(target_b_pos + UP * 0.9),
            run_time=2,
            rate_func=linear
        )
        
        # Collision 1: B hits A
        self.play(
            block_a.animate.set_color(YELLOW),
            collision_count.animate.set_value(1),
            run_time=0.1
        )
        self.play(block_a.animate.set_color(BLUE), run_time=0.1)
        
        # Block A moves to wall
        # Wall at x=0.5. A center x = wall x + 0.3 = 0.8
        target_a_x = 0.8
        target_a_pos = np.array([target_a_x, self.grid["E2"][1], 0])
        
        self.play(
            block_a.animate.move_to(target_a_pos),
            label_a.animate.move_to(target_a_pos + UP * 0.7),
            run_time=0.8,
            rate_func=linear
        )
        
        # Collision 2: A hits wall
        self.play(
            block_a.animate.set_color(YELLOW),
            collision_count.animate.set_value(2),
            run_time=0.1
        )
        self.play(block_a.animate.set_color(BLUE), run_time=0.1)
        
        self.wait(2)
        self.lecture[2].set_color(WHITE)
