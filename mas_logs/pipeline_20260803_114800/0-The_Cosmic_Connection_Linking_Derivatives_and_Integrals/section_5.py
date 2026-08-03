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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "We don't need infinite sums to find areas.",
            "Find the antiderivative F whose derivative is f.",
            "The integral is simply the change in F.",
            "Evaluate F at the boundaries b and a.",
            "This shortcut makes complex calculations simple."
        ]
        self.setup_layout("The Calculation Shortcut", lecture_lines)

        # --- Setup Axes and Curve ---
        axes = Axes(
            x_range=[0, 5],
            y_range=[0, 5],
            x_length=3,
            y_length=3,
            axis_config={"color": WHITE, "include_tip": False}
        )
        self.place_in_area(axes, "A2", "D5")
        
        def func(x):
            return 0.2 * (x - 1) * (x - 4)**2 + 1
        
        curve = axes.plot(func, x_range=[0.5, 4.5], color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Yellow Riemann Rectangles (#FFFF00)
        rects = axes.get_riemann_rectangles(curve, x_range=[1, 4], dx=0.2, color="#FFFF00", fill_opacity=0.5)
        
        self.play(Create(axes), Create(curve))
        self.play(Create(rects))
        self.wait(1)
        self.play(FadeOut(rects))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FF8C00") # Orange
        
        antiderivative_relation = MathTex("F'(x) = f(x)", color="#FF8C00")
        self.place_at_grid(antiderivative_relation, "E1", scale_factor=0.8)
        
        self.play(Write(antiderivative_relation))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(WHITE)
        
        integral_notation = MathTex(r"\int_a^b f(x) dx = \Delta F", color=WHITE)
        self.place_in_area(integral_notation, "E2", "E5", scale_factor=0.8)
        
        self.play(Write(integral_notation))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FF8C00") # Orange
        
        # a and b positions
        a_val, b_val = 1, 4
        p1_top = axes.c2p(a_val, func(a_val))
        p2_top = axes.c2p(b_val, func(b_val))
        p1_base = axes.c2p(a_val, 0)
        p2_base = axes.c2p(b_val, 0)
        
        # Asset: Orange pillars [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/pillar.svg]
        pillar_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/pillar.svg"
        pillar_a = SVGMobject(pillar_path, color="#FF8C00")
        pillar_b = SVGMobject(pillar_path, color="#FF8C00")
        
        # Scale pillars to fit the height of the curve
        h_a = p1_top[1] - p1_base[1]
        h_b = p2_top[1] - p2_base[1]
        
        pillar_a.stretch_to_fit_height(h_a)
        pillar_a.stretch_to_fit_width(0.3)
        pillar_a.move_to(p1_base + UP * (h_a / 2))
        
        pillar_b.stretch_to_fit_height(h_b)
        pillar_b.stretch_to_fit_width(0.3)
        pillar_b.move_to(p2_base + UP * (h_b / 2))
        
        label_fa = MathTex("F(a)", color="#FF8C00", font_size=24)
        label_fb = MathTex("F(b)", color="#FF8C00", font_size=24)
        
        label_fa.next_to(pillar_a, UP, buff=0.1)
        label_fb.next_to(pillar_b, UP, buff=0.1)
        
        self.play(DrawBorderThenFill(pillar_a), DrawBorderThenFill(pillar_b))
        self.play(Write(label_fa), Write(label_fb))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(WHITE)
        
        shortcut_formula = MathTex(r"F(b) - F(a)", color=WHITE)
        self.place_at_grid(shortcut_formula, "F4", scale_factor=1.0)
        
        # Connect pillars visual
        connection_line = DashedLine(p1_top, p2_top, color=WHITE)
        
        self.play(Create(connection_line))
        self.play(Write(shortcut_formula))
        
        # Update integral notation to show the result
        final_relation = MathTex(r"\int_a^b f(x) dx = F(b) - F(a)", color=WHITE)
        self.place_in_area(final_relation, "E2", "E5", scale_factor=0.8)
        
        self.play(Transform(integral_notation, final_relation))
        self.wait(2)
