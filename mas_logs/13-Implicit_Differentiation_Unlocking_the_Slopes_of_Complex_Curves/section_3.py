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

class Section3Scene(TeachingScene):
    def construct(self):
        # Initialization
        # Updated to 3 lines based on snapshot
        lines = [
            "First, differentiate both sides with respect to x.",
            "Attach a dy/dx tag to every y-derivative.",
            "Use algebra to isolate dy/dx on one side."
        ]
        self.setup_layout("The Step-by-Step Algorithm", lines)
        
        # Colors
        COLOR_DX = YELLOW_A
        COLOR_DYDX = "#FF69B4"  # Pink Sticky Note color
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Balance scale construction
        beam = Line(LEFT, RIGHT, color=GRAY).scale(2.5)
        pivot = Triangle(color=GRAY).scale(0.3).rotate(PI).next_to(beam, DOWN, buff=0)
        scale = VGroup(beam, pivot)
        self.place_in_area(scale, "D1", "D6")
        
        # Initial Equation
        eq1_left = Text("x^2 + y^2", font_size=36)
        eq1_right = Text("25", font_size=36)
        
        self.place_at_grid(eq1_left, "C2")
        self.place_at_grid(eq1_right, "C5")
        
        # Operator d/dx
        ddx_left = Text("d/dx", color=COLOR_DX, font_size=36)
        ddx_right = Text("d/dx", color=COLOR_DX, font_size=36)
        
        ddx_left.next_to(eq1_left, LEFT, buff=0.1)
        ddx_right.next_to(eq1_right, LEFT, buff=0.1)

        self.play(Create(scale))
        self.play(Write(eq1_left), Write(eq1_right))
        self.wait(0.5)
        self.play(FadeIn(ddx_left), FadeIn(ddx_right))
        self.wait(1)

        # Result of differentiation - standardized to Row C (Fixing Issues 37, 38)
        res_x = Text("2x", font_size=36)
        res_plus = Text("+", font_size=36)
        res_y = Text("2y", font_size=36)
        res_eq = Text("=", font_size=36)
        res_zero = Text("0", font_size=36)
        
        res_group = VGroup(res_x, res_plus, res_y, res_eq, res_zero).arrange(RIGHT, buff=0.2)
        # Fix: Standardize to C2-C5
        self.place_in_area(res_group, "C2", "C5")
        
        self.play(
            ReplacementTransform(VGroup(ddx_left, eq1_left), VGroup(res_x, res_plus, res_y)),
            ReplacementTransform(VGroup(ddx_right, eq1_right), VGroup(res_eq, res_zero)),
            FadeOut(scale)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # The Pink Sticky Note dy/dx
        sticky_note_bg = SurroundingRectangle(res_y, color=COLOR_DYDX, fill_opacity=0.3, fill_color=COLOR_DYDX, buff=0.1)
        dydx_tag = Text("dy/dx", color=COLOR_DYDX, font_size=24).next_to(res_y, RIGHT, buff=0.1)
        
        self.play(Create(sticky_note_bg), Write(dydx_tag))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Re-arrange: 2y dy/dx = -2x
        l4_term1 = res_y.copy()
        l4_tag = dydx_tag.copy()
        l4_eq = Text("=", font_size=36)
        l4_term2 = Text("-2x", font_size=36)
        
        line4_group = VGroup(l4_term1, l4_tag, l4_eq, l4_term2).arrange(RIGHT, buff=0.2)
        # Fix: Standardize to C2-C5 (Issue 37)
        self.place_in_area(line4_group, "C2", "C5")
        
        self.play(
            ReplacementTransform(VGroup(res_x, res_plus, res_y, res_eq, res_zero, sticky_note_bg, dydx_tag), line4_group)
        )
        self.wait(1)

        # Isolate dy/dx: dy/dx = -x/y
        final_dydx = Text("dy/dx", color=COLOR_DYDX, font_size=36)
        final_eq = Text("=", font_size=36)
        final_frac = Text("-x / y", font_size=36)
        
        final_group = VGroup(final_dydx, final_eq, final_frac).arrange(RIGHT, buff=0.2)
        # Fix: Standardize to C2-C5 (Issue 37, 39)
        self.place_in_area(final_group, "C2", "C5")
        
        # isolate dy/dx in a box
        result_box = SurroundingRectangle(final_group, color=COLOR_DYDX, buff=0.2)
        
        self.play(ReplacementTransform(line4_group, final_group))
        self.play(Create(result_box))
        self.wait(2)
