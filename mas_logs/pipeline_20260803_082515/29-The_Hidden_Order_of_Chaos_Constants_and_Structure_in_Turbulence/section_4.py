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
        lecture_lines = [
            "Kolmogorov discovered universal scaling in the inertial range.",
            "The energy spectrum follows a specific power law.",
            "Energy density is proportional to wavenumber to -5/3.",
            "The Kolmogorov constant remains remarkably universal.",
            "This law forms the spine of turbulence theory."
        ]
        self.setup_layout("The Mathematical Spine: The -5/3 Law", lecture_lines)
        
        # Colors
        COLOR_WHITE = "#FFFFFF"
        COLOR_CYAN = "#00FFFF"
        COLOR_LIME = "#00FF00"
        COLOR_DIM = "#444444"

        # === Animation for Lecture Line 1 ===
        # Kolmogorov discovered universal scaling in the inertial range.
        self.play(self.lecture[0].animate.set_color(COLOR_WHITE))
        
        # Create log-log axes representation
        axes = Axes(
            x_range=[0, 6, 1],
            y_range=[0, 6, 1],
            x_length=5,
            y_length=5,
            axis_config={"include_tip": False, "color": GREY},
        )
        # Issue 34 fix: Reduced scale_factor to 0.75 for better spacing
        self.place_in_area(axes, 'A1', 'F6', scale_factor=0.75)
        
        # Define the -5/3 slope line
        # Start at log(k)=1, log(E)=5; End at log(k)=4, log(E)=0 (Slope = -5/3)
        start_pt = axes.c2p(1, 5)
        end_pt = axes.c2p(4, 0)
        slope_line = Line(start_pt, end_pt, color=COLOR_WHITE, stroke_width=4)
        
        self.play(Create(axes), Create(slope_line))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The energy spectrum follows a specific power law.
        self.play(
            self.lecture[0].animate.set_color(COLOR_DIM),
            self.lecture[1].animate.set_color(COLOR_CYAN)
        )
        
        # Label the line with formula E(k) proportional to k^(-5/3) in cyan
        formula = MathTex(r"E(k) \propto k^{-5/3}", color=COLOR_CYAN)
        # Issue 35 fix: Shifted formula to B5 and reduced scale to 0.8
        self.place_at_grid(formula, 'B5', scale_factor=0.8)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Energy density is proportional to wavenumber to -5/3.
        self.play(
            self.lecture[1].animate.set_color(COLOR_DIM),
            self.lecture[2].animate.set_color(COLOR_CYAN)
        )
        
        # Pulse the 'Inertial Subrange' section
        subrange_label = Text("Inertial Subrange", font_size=18, color=COLOR_WHITE)
        # Issue 36 fix: Centered label across D3-D5 with 0.8 scale
        self.place_in_area(subrange_label, 'D3', 'D5', scale_factor=0.8)
        
        # Section of the line to highlight
        pulse_line = Line(axes.c2p(1.5, 4.16), axes.c2p(3.5, 0.83), color=COLOR_CYAN, stroke_width=8)
        
        self.play(Write(subrange_label))
        self.play(FadeIn(pulse_line))
        self.play(pulse_line.animate.set_stroke(width=14), run_time=0.4)
        self.play(pulse_line.animate.set_stroke(width=8), run_time=0.4)
        self.play(FadeOut(pulse_line))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The Kolmogorov constant remains remarkably universal.
        self.play(
            self.lecture[2].animate.set_color(COLOR_DIM),
            self.lecture[3].animate.set_color(COLOR_LIME)
        )
        
        # Show the Kolmogorov constant C_K in lime
        ck_const = MathTex(r"C_K \approx 1.5", color=COLOR_LIME)
        self.place_at_grid(ck_const, 'B2', scale_factor=0.9)
        
        self.play(Write(ck_const))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This law forms the spine of turbulence theory.
        self.play(
            self.lecture[3].animate.set_color(COLOR_DIM),
            self.lecture[4].animate.set_color(COLOR_WHITE)
        )
        
        # A digital ruler aligns with the -5/3 slope
        ruler = VGroup()
        ruler_rect = Rectangle(width=3, height=0.3, color=COLOR_WHITE, fill_opacity=0.3)
        ticks = VGroup(*[
            Line(UP*0.1, DOWN*0.1, color=COLOR_WHITE).move_to(LEFT*1.5 + RIGHT*i*0.5) 
            for i in range(7)
        ])
        ruler_label = Text("-5/3", font_size=16, color=COLOR_WHITE).move_to(ruler_rect.get_center())
        ruler.add(ruler_rect, ticks, ruler_label)
        
        # Position ruler parallel to the line
        angle = np.arctan2(end_pt[1] - start_pt[1], end_pt[0] - start_pt[0])
        ruler.rotate(angle)
        # Offset start position for animation
        ruler.move_to(slope_line.get_center() + UP*0.8 + LEFT*0.5)
        
        self.play(FadeIn(ruler))
        self.play(ruler.animate.move_to(slope_line.get_center()))
        self.wait(3)
