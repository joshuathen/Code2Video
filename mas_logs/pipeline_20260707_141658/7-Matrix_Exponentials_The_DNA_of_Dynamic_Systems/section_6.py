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
        # Data from storyboard
        title = "Application: The Predator-Prey Ecosystem"
        lines = [
            "Apply the matrix exponential to a predator-prey system.",
            "The matrix A encodes how species interact and grow.",
            "Solve for future populations using the exponential operator."
        ]
        
        self.setup_layout(title, lines)
        
        # Colors
        GOLD = "#FFD700"
        RABBIT_COLOR = "#FFFFFF"
        FOX_COLOR = "#FF8C00"
        
        # === Animation for Lecture Line 1 ===
        # Show dV/dt = AV where V = [R, F]
        # Use rabbit icon for R
        
        # Equation: dV/dt = A V
        eq_system = Text("dV/dt = A V", font_size=36, color=WHITE)
        self.place_in_area(eq_system, "B1", "B6", scale_factor=1.0) # Issue 36
        
        # Components of V
        # Asset: [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/rabbit.svg]
        rabbit_icon = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/rabbit.svg")
        rabbit_icon.set_color(WHITE).scale(0.3)
        
        fox_text = Text("F", font_size=36, color=FOX_COLOR)
        
        v_elements = VGroup(rabbit_icon, fox_text).arrange(DOWN, buff=0.4)
        l_bracket_v = Text("[", font_size=60).stretch_to_fit_height(v_elements.get_height() + 0.2).next_to(v_elements, LEFT, buff=0.1)
        r_bracket_v = Text("]", font_size=60).stretch_to_fit_height(v_elements.get_height() + 0.2).next_to(v_elements, RIGHT, buff=0.1)
        vector_v = VGroup(l_bracket_v, v_elements, r_bracket_v)
        
        v_label = Text("V = ", font_size=32).next_to(vector_v, LEFT)
        v_group = VGroup(v_label, vector_v)
        self.place_at_grid(v_group, "C2", scale_factor=0.8)
        
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(Write(eq_system))
        self.play(FadeIn(v_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The matrix A encodes interactions. Highlight off-diagonal in Gold.
        
        # Construct Matrix A: [[0.1, -0.2], [0.1, -0.1]]
        a00 = Text("0.1", font_size=30)
        a01 = Text("-0.2", font_size=30, color=GOLD) # Interaction term
        a10 = Text("0.1", font_size=30, color=GOLD)  # Interaction term
        a11 = Text("-0.1", font_size=30)
        
        matrix_elements = VGroup(
            VGroup(a00, a10).arrange(DOWN, buff=0.5),
            VGroup(a01, a11).arrange(DOWN, buff=0.5)
        ).arrange(RIGHT, buff=0.8)
        
        l_bracket_a = Text("[", font_size=60).stretch_to_fit_height(matrix_elements.get_height() + 0.2).next_to(matrix_elements, LEFT, buff=0.1)
        r_bracket_a = Text("]", font_size=60).stretch_to_fit_height(matrix_elements.get_height() + 0.2).next_to(matrix_elements, RIGHT, buff=0.1)
        matrix_a = VGroup(l_bracket_a, matrix_elements, r_bracket_a)
        
        a_label = Text("A = ", font_size=32).next_to(matrix_a, LEFT)
        matrix_group = VGroup(a_label, matrix_a)
        self.place_in_area(matrix_group, "D1", "F6", scale_factor=0.8) # Issue 37
        
        self.play(self.lecture[1].animate.set_color(GOLD))
        self.play(FadeIn(matrix_group))
        self.play(Indicate(a01), Indicate(a10), color=GOLD)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Solve for future populations using exponential operator. Plot curves.
        
        self.play(self.lecture[2].animate.set_color(WHITE))
        
        # Solution formula: V(t) = e^{At} V(0)
        sol_text = Text("V(t) = exp(At) V(0)", font_size=32, color=WHITE)
        self.place_at_grid(sol_text, "A4", scale_factor=1.0)
        
        # Create Axes for population plot
        axes = Axes(
            x_range=[0, 10, 2],
            y_range=[-1.5, 1.5, 1],
            axis_config={"include_tip": True, "font_size": 24},
            x_length=4,
            y_length=2.5
        )
        self.place_in_area(axes, "D1", "F6", scale_factor=0.7) # Repositioning to show graph
        
        # Oscillating curves
        # R(t) = cos(t) + sin(t)
        # F(t) = sin(t)
        curve_r = axes.plot(lambda t: np.cos(t), color=RABBIT_COLOR)
        curve_f = axes.plot(lambda t: np.sin(t), color=FOX_COLOR)
        
        label_r = Text("Rabbits", font_size=18, color=RABBIT_COLOR).next_to(axes, UP, buff=0.1).shift(LEFT)
        label_f = Text("Foxes", font_size=18, color=FOX_COLOR).next_to(axes, UP, buff=0.1).shift(RIGHT)
        
        # Clear previous matrix to show graph
        self.play(FadeOut(matrix_group), FadeOut(v_group))
        self.play(Write(sol_text))
        self.play(Create(axes), Write(label_r), Write(label_f))
        self.play(Create(curve_r), Create(curve_f), run_time=3)
        self.wait(2)
