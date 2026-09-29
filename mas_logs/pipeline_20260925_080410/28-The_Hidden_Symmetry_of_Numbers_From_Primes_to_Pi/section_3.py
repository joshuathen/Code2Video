from manim import *
import os

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
        lecture_lines = [
            "The Prime Number Theorem defines their density.", 
            "As numbers grow, prime density decreases.", 
            "This reveals a link to the constant pi."
        ]
        self.setup_layout("Bridging to Pi: The PNT", lecture_lines)
        
        # Load Assets
        calc_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        compass_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        
        PNT_formula = MathTex(r"\pi(x) \approx \frac{x}{\ln(x)}", color="#FFD700")
        circle = Circle(radius=1.5, color="#00FFFF")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        # Fix for issue 26: PNT_formula positioned at 'A2'-'A6'
        self.place_in_area(PNT_formula, 'A2', 'A6', scale_factor=0.9)
        self.place_at_grid(calc_icon, "A1", scale_factor=0.5)
        self.play(FadeIn(calc_icon), Write(PNT_formula))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        # Fix for issue 28: dots positioned at 'D1'-'F3'
        dots = VGroup(*[Dot(radius=0.05, color="#FFFFFF") for _ in range(20)])
        self.place_in_area(dots, "D1", "F3", scale_factor=0.7)
        self.play(FadeIn(dots))
        self.play(dots.animate.set_opacity(0.3))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        # Fix for issue 27: circle positioned at 'E5'
        self.place_at_grid(circle, "E5", scale_factor=0.8)
        self.place_at_grid(compass_icon, "F6", scale_factor=0.5)
        self.play(Create(circle), FadeIn(compass_icon))
        self.wait(1)
