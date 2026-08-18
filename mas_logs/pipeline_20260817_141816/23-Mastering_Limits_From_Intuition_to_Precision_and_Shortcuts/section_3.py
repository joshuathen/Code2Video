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
        self.setup_layout("L'Hôpital's Rule: The Shortcut for Indeterminate Forms", 
                          ["Algebra fails for indeterminate forms like 0/0.", 
                           "L'Hôpital's Rule uses derivatives as a shortcut.", 
                           "We compare slopes of numerator and denominator."])
        
        # === Animation for Lecture Line 1 ===
        # Algebra fails for indeterminate forms like 0/0.
        frac = MathTex(r"\frac{0}{0}", font_size=72)
        self.place_at_grid(frac, 'B3')
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        try:
            icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
            self.place_at_grid(icon, 'B5', scale_factor=0.5)
        except:
            icon = Circle(radius=0.3, color=RED).move_to(self.grid['B5'])

        self.play(FadeIn(frac), FadeIn(icon), self.lecture[0].animate.set_color("#FF4500"))
        self.play(Indicate(frac, color="#FF4500"), run_time=2)

        # === Animation for Lecture Line 2 ===
        # L'Hôpital's Rule uses derivatives as a shortcut.
        frac_deriv = MathTex(r"\frac{f'(x)}{g'(x)}", font_size=72)
        self.place_at_grid(frac_deriv, 'B3')
        
        self.play(ReplacementTransform(frac, frac_deriv), self.lecture[1].animate.set_color("#00FFFF"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # We compare slopes of numerator and denominator.
        tangent1 = Line(start=LEFT, end=RIGHT, color="#FFFF00").scale(0.5)
        tangent2 = Line(start=LEFT, end=RIGHT, color="#00FF00").scale(0.5)
        
        self.place_at_grid(tangent1, 'D2')
        self.place_at_grid(tangent2, 'D4')
        
        self.play(Create(tangent1), Create(tangent2), self.lecture[2].animate.set_color("#00FF00"))
        self.wait(2)
