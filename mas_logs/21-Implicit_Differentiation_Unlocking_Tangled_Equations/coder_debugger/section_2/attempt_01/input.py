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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "Prerequisite: The Secret Identity of 'y'", 
            [
                "Imagine y as a mysterious box.",
                "Inside the box is a function of x.",
                "Differentiating y requires the chain rule.",
                "Multiply by the derivative of the inside.",
                "This leaves us with dy/dx."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Line 0: "Imagine y as a mysterious box." Color: #FF00FF
        self.lecture[0].set_color("#FF00FF")
        
        # Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/box.svg
        box_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/box.svg")
        box_svg.set_color("#FF00FF")
        self.place_in_area(box_svg, "B2", "C3", scale_factor=1.4)
        
        y_label = Text("y", color="#FF00FF", font_size=50)
        self.place_in_area(y_label, "B2", "C3")
        
        func_label = Text("y is a function of x", font_size=24, color="#FF00FF")
        # VideoCritic fix: place func_label in area A2-A3
        self.place_in_area(func_label, "A2", "A3", scale_factor=0.8)
        
        self.play(DrawBorderThenFill(box_svg), Write(y_label), Write(func_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 1: "Inside the box is a function of x." Color: #FF00FF
        self.lecture[1].set_color("#FF00FF")
        
        # Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/doll.svg
        doll_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/doll.svg")
        doll_svg.set_color("#FF00FF")
        self.place_in_area(doll_svg, "B2", "C3", scale_factor=1.2)
        
        self.play(
            box_svg.animate.scale(1.2).set_opacity(0.3),
            FadeOut(y_label),
            FadeIn(doll_svg),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 2: "Differentiating y requires the chain rule." Color: #FFA500
        self.lecture[2].set_color("#FFA500")
        
        # VideoCritic fix: d_dx_term at D1
        d_dx_term = Text("d/dx [y³]", font_size=36)
        self.place_at_grid(d_dx_term, "D1", scale_factor=0.9)
        
        # VideoCritic fix: equals at D2
        equals = Text("=", font_size=36)
        self.place_at_grid(equals, "D2", scale_factor=1.0)
        
        # VideoCritic fix: outer_deriv at D3
        outer_deriv = Text("3y²", color="#FFA500", font_size=40)
        self.place_at_grid(outer_deriv, "D3", scale_factor=1.0)
        
        self.play(Write(d_dx_term))
        self.play(Write(equals), Write(outer_deriv))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Line 3: "Multiply by the derivative of the inside." Color: #FF0000
        self.lecture[3].set_color("#FF0000")
        
        # VideoCritic fix: inner_deriv at D4
        inner_deriv = Text("· dy/dx", color="#FF0000", font_size=40)
        self.place_at_grid(inner_deriv, "D4", scale_factor=1.0)
        
        self.play(Write(inner_deriv))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Line 4: "This leaves us with dy/dx." Color: #FF0000
        self.lecture[4].set_color("#FF0000")
        
        # Full result highlight
        full_result = VGroup(outer_deriv, inner_deriv)
        glow = SurroundingRectangle(full_result, color=YELLOW, buff=0.1).set_stroke(width=2)
        
        self.play(Create(glow))
        self.play(Indicate(full_result, color=YELLOW, scale_factor=1.1))
        self.wait(2)
