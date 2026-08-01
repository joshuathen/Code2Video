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

class Section4Scene(TeachingScene):
    def construct(self):
        # Data from storyboard and outline
        title_text = "The Boundary of Chaos: Julia Sets"
        lecture_lines = [
            "The Julia set is the boundary of these behaviors.",
            "It separates points that stay from points that escape.",
            "On this edge, movement is unpredictable and chaotic.",
            "Zooming in reveals infinite, self-similar fractal patterns.",
            "This boundary contains the essence of holomorphic dynamics."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Hex colors as per storyboard
        COLOR_C = ManimColor("#FFFF00")
        COLOR_DARK_BLUE = ManimColor("#00008B")
        COLOR_CYAN = ManimColor("#00FFFF")
        COLOR_BOUNDARY = ManimColor("#FFFFFF")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_C)
        c_val = MathTex("c = -0.4 + 0.6i", color=COLOR_C)
        # Issue 32: Fix: Line 69: self.place_at_grid(c_val, 'B4', scale_factor=0.8)
        self.place_at_grid(c_val, 'B4', scale_factor=0.8)
        self.play(Write(c_val))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_CYAN)
        
        # Grid parameters for the fractal area
        res = 18 # Balanced resolution for render time
        complex_grid = VGroup()
        c_complex = complex(-0.4, 0.6)
        
        # Create squares for pixelated effect
        for row in range(res):
            for col in range(res):
                sq = Square(side_length=1.0, stroke_width=0.5, stroke_color=WHITE)
                
                re = -1.5 + (col / res) * 3.0
                im = 1.5 - (row / res) * 3.0
                z = complex(re, im)
                
                escape_iter = 0
                max_iter = 20
                for _ in range(max_iter):
                    if abs(z) > 2:
                        break
                    z = z**2 + c_complex
                    escape_iter += 1
                
                sq.escape_iter = escape_iter
                sq.max_iter = max_iter
                complex_grid.add(sq)

        complex_grid.arrange_in_grid(rows=res, cols=res, buff=0)
        # Issue 31: Fix: Line 75: self.place_in_area(complex_grid, 'C3', 'F6', scale_factor=0.9)
        self.place_in_area(complex_grid, 'C3', 'F6', scale_factor=0.9)
        
        self.play(FadeIn(complex_grid, lag_ratio=0.01), run_time=1.5)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_CYAN)
        
        # Create julia_fractal (filled version) from complex_grid
        julia_fractal_visual = complex_grid.copy()
        for sq in julia_fractal_visual:
            if sq.escape_iter == sq.max_iter:
                target_color = BLACK
            else:
                target_color = interpolate_color(COLOR_DARK_BLUE, COLOR_CYAN, sq.escape_iter / sq.max_iter)
            sq.set_fill(target_color, opacity=1)
            sq.set_stroke(width=0)

        # Transition to colored fractal
        self.play(Transform(complex_grid, julia_fractal_visual), run_time=2)
        # Link variable names to the visible object for subsequent animations
        julia_fractal = complex_grid 
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(COLOR_BOUNDARY)
        
        # Highlight boundary (where escape_iter is between 5 and 19)
        boundary_sqs = VGroup(*[sq for sq in julia_fractal if 5 < sq.escape_iter < 20])
        if len(boundary_sqs) > 0:
            # Highlight subset to avoid performance lag while demonstrating complexity
            self.play(
                AnimationGroup(*[Indicate(sq, color=COLOR_BOUNDARY) for sq in boundary_sqs[::4]], lag_ratio=0.05),
                run_time=2
            )

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(COLOR_BOUNDARY)
        
        # Zoom into boundary section
        zoom_target_point = boundary_sqs[len(boundary_sqs)//3].get_center() if len(boundary_sqs) > 0 else julia_fractal.get_center()
        
        # Perform zoom and handle potential obstruction by re-constraining the area
        # Issue 30: Fix: Line 82: self.place_in_area(julia_fractal, 'B3', 'F6', scale_factor=0.7)
        self.play(
            julia_fractal.animate.scale(4, about_point=zoom_target_point),
            c_val.animate.move_to(self.grid["B1"]).scale(0.8),
            run_time=2
        )
        
        # Ensure final state is constrained to the allowed right-side area
        self.play(
            self.place_in_area(julia_fractal, 'B3', 'F6', scale_factor=0.7).animate,
            run_time=1
        )
        
        self.wait(2)
