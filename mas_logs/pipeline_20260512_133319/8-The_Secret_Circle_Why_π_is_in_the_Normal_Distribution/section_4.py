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
        # Initial Setup with prompt-mandated lecture lines
        lines = [
            'Rectangular coordinates make circular symmetry difficult to solve.',
            'We transform the entire plane into polar coordinates.',
            'This shift introduces an essential scaling factor, r.'
        ]
        self.setup_layout("Switching to Polar Coordinates", lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Create a rectangular grid (green #83C167)
        cart_grid = VGroup()
        for i in range(-3, 4):
            line_h = Line(start=[-1.5, i/2, 0], end=[1.5, i/2, 0], color="#83C167", stroke_width=1)
            line_v = Line(start=[i/2, -1.5, 0], end=[i/2, 1.5, 0], color="#83C167", stroke_width=1)
            cart_grid.add(line_h, line_v)
        
        # FIX Issue 35: scale_factor set to 1.0
        self.place_in_area(cart_grid, "C2", "E5", scale_factor=1.0)
        
        # Small rectangular patch (green)
        patch_rect = Rectangle(width=0.5, height=0.5, color="#83C167", fill_opacity=0.5)
        # Position relative to cartesian grid center
        patch_rect.move_to(cart_grid.get_center() + RIGHT*0.5 + UP*0.5)
        
        self.play(Create(cart_grid), FadeIn(patch_rect))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Create a polar grid
        polar_grid = VGroup()
        for r in range(1, 5):
            circle = Circle(radius=r/2.5, color="#83C167", stroke_width=1)
            polar_grid.add(circle)
        for angle in np.linspace(0, 2*np.pi, 13):
            line = Line(start=[0,0,0], end=[1.6*np.cos(angle), 1.6*np.sin(angle), 0], color="#83C167", stroke_width=1)
            polar_grid.add(line)
        
        self.place_in_area(polar_grid, "C2", "E5", scale_factor=1.0)
        
        # Polar wedge patch
        patch_wedge = AnnularSector(
            inner_radius=0.8, outer_radius=1.2, 
            angle=30*DEGREES, start_angle=45*DEGREES,
            color=YELLOW, fill_opacity=0.5
        )
        # Position relative to polar grid center
        patch_wedge.move_to(polar_grid.get_center() + RIGHT*0.8 + UP*0.8)

        # Formula 1: x² + y² = r² (Yellow #FFFF00)
        formula_1 = Text("x² + y² = r²", color="#FFFF00", font_size=32)
        # FIX Issue 36: Align formula_1 horizontally using area B2-B5
        self.place_in_area(formula_1, "B2", "B5", scale_factor=0.9)
        
        self.play(
            ReplacementTransform(cart_grid, polar_grid),
            ReplacementTransform(patch_rect, patch_wedge),
            Write(formula_1)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Formula 2: dx dy = r dr dθ
        # Using components to highlight 'r' later
        f2_part1 = Text("dx dy = ", color="#FFFF00", font_size=32)
        f2_r = Text("r", color="#FFFF00", font_size=36, weight=BOLD)
        f2_part2 = Text(" dr dθ", color="#FFFF00", font_size=32)
        formula_2 = VGroup(f2_part1, f2_r, f2_part2).arrange(RIGHT, buff=0.1)
        
        # FIX Issue 37: Align formula_2 horizontally using area F2-F5
        self.place_in_area(formula_2, "F2", "F5", scale_factor=0.9)
        
        self.play(Write(formula_2))
        
        # Visualization of Jacobian factor
        # Pulse the wedge to show it's larger further out
        self.play(patch_wedge.animate.scale(1.1).set_color(ORANGE), run_time=0.5)
        self.play(patch_wedge.animate.scale(1/1.1).set_color(YELLOW), run_time=0.5)
        
        # Highlight the 'r'
        circle_r = Circle(radius=0.3, color=WHITE).move_to(f2_r.get_center())
        
        self.play(Create(circle_r))
        self.play(f2_r.animate.scale(1.5).set_color(WHITE), run_time=0.5)
        self.play(f2_r.animate.scale(1/1.5).set_color("#FFFF00"), run_time=0.5)
        self.play(FadeOut(circle_r))
        
        self.wait(2)
