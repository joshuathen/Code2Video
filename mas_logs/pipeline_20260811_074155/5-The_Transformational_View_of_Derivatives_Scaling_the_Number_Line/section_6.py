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
        # LECTURE CONTENT
        lecture_lines = [
            "The equation df equals derivative times dx.",
            "Derivatives convert input changes into output changes.",
            "It measures how much we stretch the number line.",
            "Compare the slope view with this magnification view.",
            "See derivatives as the local stretch of mathematical fabric."
        ]
        self.setup_layout("Summary: The Derivative as a Local Magnifier", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # The equation df equals derivative times dx.
        self.lecture[0].set_color(WHITE)
        formula = MathTex(r"df = f'(x) \cdot dx", font_size=42, color=WHITE)
        self.place_in_area(formula, "A2", "A5")
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Derivatives convert input changes into output changes.
        self.lecture[1].set_color(TEAL)
        
        # Slope Graph (Left Side)
        axes = Axes(
            x_range=[0, 3, 1], y_range=[0, 3, 1], 
            x_length=2.5, y_length=2.5,
            axis_config={"include_tip": False, "color": BLUE_E}
        )
        curve = axes.plot(lambda x: 0.3 * x**2, x_range=[0, 3], color=TEAL)
        p_x, p_y = 1.5, 0.3 * 1.5**2
        slope_val = 0.6 * 1.5
        dot_slope = Dot(axes.c2p(p_x, p_y), color=YELLOW)
        tan_line = Line(
            axes.c2p(p_x - 0.7, p_y - 0.7 * slope_val),
            axes.c2p(p_x + 0.7, p_y + 0.7 * slope_val),
            color=WHITE
        )
        slope_group = VGroup(axes, curve, dot_slope, tan_line)
        self.place_in_area(slope_group, "B1", "D3", scale_factor=0.8)
        
        slope_label = Text("Slope View", font_size=20, color=WHITE)
        self.place_at_grid(slope_label, "E2")

        # Scaling View (Right Side)
        l_in = NumberLine(x_range=[0, 4, 1], length=2.5, color=BLUE, include_numbers=True, font_size=16)
        l_out = NumberLine(x_range=[0, 4, 1], length=2.5, color=RED, include_numbers=True, font_size=16)
        scaling_lines = VGroup(l_in, l_out).arrange(DOWN, buff=1.2)
        dot_in = Dot(l_in.n2p(1.5), color=BLUE)
        dot_out = Dot(l_out.n2p(2.5), color=RED)
        map_arrow = Arrow(dot_in.get_bottom(), dot_out.get_top(), color=WHITE, buff=0.1)
        
        scaling_group = VGroup(scaling_lines, dot_in, dot_out, map_arrow)
        self.place_in_area(scaling_group, "B4", "D6", scale_factor=0.8)
        
        scaling_label = Text("Scaling View", font_size=20, color=WHITE)
        self.place_at_grid(scaling_label, "E5")

        self.play(
            Create(slope_group),
            FadeIn(slope_label),
            Create(scaling_group),
            FadeIn(scaling_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # It measures how much we stretch the number line.
        self.lecture[2].set_color(YELLOW)
        
        # Visualizing the stretch
        dx_line = Line(l_in.n2p(1.3), l_in.n2p(1.7), color=TEAL, stroke_width=8)
        df_line = Line(l_out.n2p(2.1), l_out.n2p(2.9), color=YELLOW, stroke_width=8)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/magn.svg]
        magnifier_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magn.svg")
        magnifier_svg.set_color(YELLOW)
        magnifier_text = Text("Local Magnifier", font_size=22, color=YELLOW)
        magnifier_ui = VGroup(magnifier_svg, magnifier_text).arrange(RIGHT, buff=0.2)
        
        # Issue 34, 35: place_in_area('E3', 'E4')
        self.place_in_area(magnifier_ui, "E3", "E4", scale_factor=0.6)
        
        self.play(
            Create(dx_line),
            Create(df_line),
            FadeIn(magnifier_ui)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Compare the slope view with this magnification view.
        self.lecture[3].set_color(PURPLE)
        
        everything_scaling = VGroup(scaling_group, scaling_label, magnifier_ui, dx_line, df_line)
        
        # Target area for centered scaling view (between cols 2 and 5)
        target_center = self.place_in_area(VGroup(), "B2", "D5").get_center()
        
        self.play(
            FadeOut(slope_group),
            FadeOut(slope_label),
            everything_scaling.animate.move_to(target_center)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # See derivatives as the local stretch of mathematical fabric.
        self.lecture[4].set_color(GREEN)
        
        final_summary = Text("Derivative = Local Scaling Factor", font_size=26, color=GREEN)
        # Issue 36: place_in_area('F1', 'F6')
        self.place_in_area(final_summary, "F1", "F6")
        
        self.play(Write(final_summary))
        self.play(final_summary.animate.scale(1.1), rate_func=there_and_back)
        self.wait(2)
