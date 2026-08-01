from manim import *

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
        title_text = "The Jump to Continuous: Introducing Convolution"
        lecture_lines = [
            "For continuous variables, sums transform into integrals.",
            "This merging process is defined as a convolution.",
            "The formula integrates the product of two functions.",
            "One distribution is evaluated at z minus x.",
            "This represents the remaining value to reach sum z."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        COLOR_HIGHLIGHT = "#FFFF00"
        COLOR_MAIN = "#FFFFFF"
        COLOR_LECTURE_HIGHLIGHT = "#58ACFA" # Light blue for highlighting lecture line
        
        # === Animation for Lecture Line 1 ===
        # For continuous variables, sums transform into integrals.
        self.play(self.lecture[0].animate.set_color(COLOR_LECTURE_HIGHLIGHT))
        
        sigma = MathTex(r"\sum", color=COLOR_MAIN)
        self.place_in_area(sigma, 'A2', 'C5', scale_factor=1.2) # Resolved Issue 35: Reduced scale and centered
        
        integral = MathTex(r"\int", color=COLOR_MAIN)
        self.place_in_area(integral, 'A2', 'C5', scale_factor=1.2) # Resolved Issue 35: Reduced scale and centered
        
        self.play(Write(sigma))
        self.wait(1)
        self.play(ReplacementTransform(sigma, integral))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # This merging process is defined as a convolution.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_LECTURE_HIGHLIGHT)
        )
        
        # Formula: (f * g)(z) = ∫ f(x)g(z-x) dx
        formula = MathTex(
            "(f * g)(z)", "=", "\\int", "f(x)", "g(", "z-x", ")", "dx",
            color=COLOR_MAIN
        )
        # Expanded formula area - Resolved Issue 34: Expanded to full top width A1-C6
        self.place_in_area(formula, 'A1', 'C6', scale_factor=1.0)
        
        self.play(
            FadeOut(integral),
            FadeIn(formula)
        )
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        # The formula integrates the product of two functions.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_LECTURE_HIGHLIGHT)
        )
        
        # Show axes as the conceptual "workspace" for the functions
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 5, 1],
            x_length=3.5,
            y_length=2.5,
            axis_config={"include_tip": True}
        )
        # Shifted axes to the right - Resolved Issue 36: Moved to D2-F6 and reduced scale
        self.place_in_area(axes, 'D2', 'F6', scale_factor=0.8)
        self.play(Create(axes))
        self.wait(1)
        
        # === Animation for Lecture Line 4 ===
        # One distribution is evaluated at z minus x.
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(COLOR_LECTURE_HIGHLIGHT)
        )
        
        # Highlight 'z - x' in formula and show on axis
        zx_term = formula[5] # Index 5 is "z-x"
        x_val = 2.0
        y_val = 3.0
        
        x_point = axes.c2p(x_val, 0)
        zx_point = axes.c2p(0, y_val)
        
        x_marker = Dot(x_point, color=COLOR_HIGHLIGHT)
        zx_marker = Dot(zx_point, color=COLOR_HIGHLIGHT)
        
        x_label = MathTex("x", color=COLOR_HIGHLIGHT).scale(0.7)
        x_label.next_to(x_marker, DOWN, buff=0.1)
        
        zx_label = MathTex("z-x", color=COLOR_HIGHLIGHT).scale(0.7)
        zx_label.next_to(zx_marker, LEFT, buff=0.1)
        
        self.play(
            Indicate(zx_term, color=COLOR_HIGHLIGHT, scale_factor=1.3),
            FadeIn(x_marker), FadeIn(x_label),
            FadeIn(zx_marker), FadeIn(zx_label)
        )
        self.wait(1)
        
        # === Animation for Lecture Line 5 ===
        # This represents the remaining value to reach sum z.
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(COLOR_LECTURE_HIGHLIGHT)
        )
        
        # Product f(x) * g(z-x) as a highlighted vertical line
        prod_height = 2.5
        v_line = Line(
            axes.c2p(x_val, 0),
            axes.c2p(x_val, prod_height),
            color=COLOR_HIGHLIGHT,
            stroke_width=4
        )
        v_label = MathTex("f(x)g(z-x)", color=COLOR_HIGHLIGHT).scale(0.7)
        v_label.next_to(v_line, RIGHT, buff=0.1)
        
        self.play(Create(v_line), FadeIn(v_label))
        self.play(Indicate(v_line))
        
        self.wait(2)
        
        # Cleanup
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
