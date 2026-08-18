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
        # Setup layout
        title_text = "The Intuition: Different Maps, Same World"
        lecture_lines = [
            "A point's location is absolute, but its coordinates vary.",
            "Different observers use different grids to describe space.",
            "Alice sees standard tiles; Bob sees a skewed rug."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Assets
        CAT_ASSET_PATH = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png"

        # === Animation for Lecture Line 1 ===
        # Create a fixed position for the cat in the grid area.
        # Center of B1 to F6 is roughly (3.0, -0.8).
        absolute_center = np.array([3.0, -0.8, 0])
        
        # Create cat image and place it at a fixed world position (absolute)
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png]
        cat_image = ImageMobject(CAT_ASSET_PATH)
        cat_image.scale(0.3)
        # Position fixed in space relative to the grid area center
        cat_image.move_to(absolute_center + np.array([1.2, 0.6, 0])) 
        
        # Yellow glow for emphasis
        glow = Circle(radius=0.4, color=YELLOW, stroke_width=4).set_opacity(0.6)
        glow.move_to(cat_image.get_center())
        
        # Highlight line 1 and show the absolute point
        self.play(
            self.lecture[0].animate.set_color(YELLOW),
            FadeIn(cat_image),
            Create(glow),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition highlight to line 2
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Alice's setup
        alice_grid = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            background_line_style={"stroke_color": "#ADD8E6", "stroke_width": 1, "stroke_opacity": 0.5},
            axis_config={"stroke_color": "#ADD8E6", "stroke_width": 2}
        )
        # Issue 23: Move alice_grid to 'B1' to 'F6'
        self.place_in_area(alice_grid, 'B1', 'F6', scale_factor=0.6)
        
        # Issue 22: Move labels to 'A3'
        alice_label = Text("Alice (Standard)", font_size=24, color="#ADD8E6")
        self.place_at_grid(alice_label, 'A3', scale_factor=0.6)
        
        # Bob's setup (skewed rug)
        bob_grid = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            background_line_style={"stroke_color": "#90EE90", "stroke_width": 1, "stroke_opacity": 0.5},
            axis_config={"stroke_color": "#90EE90", "stroke_width": 2}
        )
        # Skew the grid to represent an alternative basis
        matrix = [[1, 0.5, 0], [0.3, 1, 0], [0, 0, 1]]
        bob_grid.apply_matrix(matrix)
        # Issue 24: Move bob_grid to 'B1' to 'F6' and scale to 0.5
        self.place_in_area(bob_grid, 'B1', 'F6', scale_factor=0.5)
        
        # Issue 22: Move labels to 'A3'
        bob_label = Text("Bob (Basis B)", font_size=24, color="#90EE90")
        self.place_at_grid(bob_label, 'A3', scale_factor=0.6)

        # Fade in Alice's perspective
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#ADD8E6"),
            FadeIn(alice_grid),
            FadeIn(alice_label),
            run_time=1.5
        )
        self.wait(2)
        
        # Transition to Bob's perspective while cat and glow remain stationary
        self.play(
            self.lecture[2].animate.set_color("#90EE90"),
            FadeOut(alice_grid),
            FadeOut(alice_label),
            FadeIn(bob_grid),
            FadeIn(bob_label),
            run_time=1.5
        )
        self.wait(2)
        
        # Final cleanup - return text to white
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            run_time=1
        )
        self.wait(2)
