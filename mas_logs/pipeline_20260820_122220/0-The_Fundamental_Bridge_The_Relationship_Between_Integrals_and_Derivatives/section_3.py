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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Integrating a rate gives net change.", "Area under the curve measures total accumulation.", "This is the Fundamental Theorem of Calculus."]
        self.setup_layout("The Fundamental Theorem of Calculus (Part 1)", lecture_lines)
        
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 3, 1], axis_config={"include_tip": True}).scale(0.5)
        curve = axes.plot(lambda x: 0.5 * x**2 - 0.5 * x + 1, x_range=[0, 4])
        # Apply layout fix for axes
        self.place_in_area(axes, 'A3', 'C6', scale_factor=0.55)
        
        # === Animation for Lecture Line 1 ===
        # Show f(x) and its integral function F(x) [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/curve.svg]
        self.lecture[0].set_color("#FFFFFF")
        curve_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/curve.svg").scale(0.5)
        self.place_at_grid(curve_icon, 'B4')
        f_label = MathTex("f(x)").next_to(curve, UP)
        self.add(axes, curve, f_label, curve_icon)
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Highlight small increment dx on the x-axis [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/increment.svg] #FF00FF
        # Show the small area increment dF = f(x)dx [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/increment.svg] #00FFFF
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FF00FF")
        
        dx_rect = Rectangle(width=0.2, height=1, color="#00FFFF").set_fill(BLUE, opacity=0.5)
        dx_rect.move_to(axes.c2p(2, 0.5))
        inc_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/increment.svg").scale(0.3)
        self.place_at_grid(inc_icon, 'B5')
        self.add(dx_rect, inc_icon)
        
        dF_label = MathTex("dF = f(x)dx", color="#00FFFF").next_to(dx_rect, UP)
        self.add(dF_label)
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Connect derivative of F(x) to f(x) [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/curve.svg] #FFFF00
        # Display fundamental theorem formula clearly [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/increment.svg] #FFFFFF
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFFF00")
        
        curve_icon2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/curve.svg").scale(0.4)
        self.place_at_grid(curve_icon2, 'C4')
        self.add(curve_icon2)
        
        ftc_formula = MathTex(r"\\int_a^b f(x) dx = F(b) - F(a)", color="#FFFFFF")
        # Apply layout fix for formula
        self.place_in_area(ftc_formula, 'C2', 'C6', scale_factor=0.9)
        inc_icon2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/increment.svg").scale(0.3)
        self.place_at_grid(inc_icon2, 'C3')
        
        self.play(Write(ftc_formula), FadeIn(inc_icon2))
        self.wait(2)
