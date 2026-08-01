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
                'Think of y as a hidden function of x.', 
                'When differentiating y terms, use the Chain Rule.', 
                'Treat y like a box containing an x expression.', 
                'The derivative of y cubed is three y squared.', 
                'Then, multiply by the inner derivative, dy dx.'
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Line 1: 'Think of y as a hidden function of x.' Color: #FF00FF
        self.lecture[0].set_color("#FF00FF")
        
        y_box_rect = Square(side_length=1.5, color="#FF00FF")
        # Fixed FileNotFoundError: 'latex' by using Text instead of MathTex
        y_box_label = Text("y", color="#FF00FF", font_size=60)
        y_box = VGroup(y_box_rect, y_box_label)
        self.place_in_area(y_box, "B2", "C3")
        
        func_label = Text("y is a function of x", font_size=24, color="#FF00FF")
        self.place_at_grid(func_label, "A2", scale_factor=0.8)
        
        self.play(Create(y_box), Write(func_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: 'When differentiating y terms, use the Chain Rule.'
        self.lecture[1].set_color(WHITE) # Default
        
        # Fixed FileNotFoundError: 'latex' by using Text instead of MathTex
        inner_content = Text("y(x)", color="#FF00FF", font_size=45)
        self.place_in_area(inner_content, "B2", "C3")
        
        self.play(
            FadeOut(y_box_label),
            FadeIn(inner_content),
            y_box_rect.animate.scale(1.2),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3: 'Treat y like a box containing an x expression.'
        self.lecture[2].set_color("#FF00FF")
        
        inner_box_rect = Square(side_length=0.8, color="#FF00FF").set_stroke(opacity=0.5)
        self.place_in_area(inner_box_rect, "B2", "C3")
        
        self.play(Create(inner_box_rect))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Line 4: 'The derivative of y cubed is three y squared.' Color: #FFA500
        self.lecture[3].set_color("#FFA500")
        
        # Fixed FileNotFoundError: 'latex' by using Text with unicode superscripts
        d_dx_term = Text("d/dx [y³]", font_size=40)
        self.place_at_grid(d_dx_term, "D2", scale_factor=1.0)
        
        outer_deriv = Text("3y²", color="#FFA500", font_size=45)
        self.place_at_grid(outer_deriv, "D4", scale_factor=1.0)
        
        equals = Text("=", font_size=40)
        self.place_at_grid(equals, "D3", scale_factor=1.0)
        
        self.play(Write(d_dx_term))
        self.play(Write(equals), Write(outer_deriv))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Line 5: 'Then, multiply by the inner derivative, dy dx.' Color: #FF0000
        self.lecture[4].set_color("#FF0000")
        
        # Fixed FileNotFoundError: 'latex' by using Text with unicode dot
        inner_deriv = Text("· dy/dx", color="#FF0000", font_size=45)
        self.place_at_grid(inner_deriv, "D5", scale_factor=1.0)
        
        # Full result highlight
        full_result = VGroup(outer_deriv, inner_deriv)
        glow = SurroundingRectangle(full_result, color=YELLOW, buff=0.1).set_stroke(width=2)
        
        self.play(Write(inner_deriv))
        self.play(Create(glow))
        self.play(Indicate(glow, color=YELLOW, scale_factor=1.1))
        self.wait(2)