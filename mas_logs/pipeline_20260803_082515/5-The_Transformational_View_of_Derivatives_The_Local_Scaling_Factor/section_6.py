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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup layout
        title_text = "Summary: The Derivative as a Local Map"
        lecture_lines = [
            "The derivative unifies slope and transformational views.",
            "It tells how space is resized at every point.",
            "Calculus is the study of local scaling."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors for lines to match animation elements
        color_1 = BLUE_B
        color_2 = RED_B
        color_3 = WHITE

        # Assets
        grid_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(color_1))
        
        # Show a complex grid being distorted by a non-linear transformation
        # Asset: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg]
        grid = SVGMobject(grid_asset_path).set_color(BLUE_B)
        # Fix from Issue 36: place_in_area(grid, 'B2', 'E5', scale_factor=0.5)
        self.place_in_area(grid, 'B2', 'E5', scale_factor=0.5)
        
        def distortion_func(p):
            # A simple non-linear mapping to show distortion
            x, y, z = p
            return np.array([
                x + 0.3 * np.sin(1.5 * y),
                y + 0.3 * np.cos(1.5 * x),
                0
            ])
        
        self.play(Create(grid))
        self.play(grid.animate.apply_function(distortion_func), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(color_2)
        )
        
        # Zoom in on a tiny portion of the grid
        # We select a region by creating a focus box
        focus_point = grid.get_center() + np.array([0.2, 0.1, 0])
        highlight_box = Square(side_length=0.4, color=RED_B).move_to(focus_point)
        
        self.play(Create(highlight_box))
        self.wait(0.5)
        
        # Create a "zoomed" version of the grid which looks uniform/linear
        # Using the same SVG but keeping it regular to represent "locally linear" scaling
        zoomed_grid = SVGMobject(grid_asset_path).set_color(BLUE_C)
        # Fix from Issue 35: place_in_area(zoomed_grid, 'B2', 'E5', scale_factor=0.6)
        self.place_in_area(zoomed_grid, 'B2', 'E5', scale_factor=0.6)
        
        # Transition: Distorted grid fades out, highlight box expands and fades, 
        # while the local uniform grid appears.
        self.play(
            FadeOut(grid),
            highlight_box.animate.scale(4).set_stroke(opacity=0),
            FadeIn(zoomed_grid),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(color_3)
        )
        
        final_text = Text("Derivative = Local Scaling Factor", color=WHITE, font_size=24)
        # Fix from Issue 34: place_in_area(final_text, 'C3', 'D6', scale_factor=0.7)
        self.place_in_area(final_text, 'C3', 'D6', scale_factor=0.7)
        
        self.play(
            FadeOut(zoomed_grid),
            Write(final_text)
        )
        self.play(Indicate(final_text, color=WHITE, scale_factor=1.1))
        self.wait(2)

        # Reset all lecture lines to white for the final frame
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
