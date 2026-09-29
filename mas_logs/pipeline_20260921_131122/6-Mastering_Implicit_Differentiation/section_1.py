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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Explicit functions isolate the variable y.", 
                         "Implicit relations keep y trapped inside.", 
                         "Circles demonstrate implicit relations perfectly."]
        self.setup_layout("The Concept: Explicit vs. Implicit", lecture_lines)
        
        eq_explicit = MathTex("y = f(x)", color=WHITE)
        eq_implicit = MathTex("x^2 + y^2 = 25", color="#00FFFF")
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg
        circle_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg")
        
        # === Animation for Lecture Line 1 ===
        # Fix for issue 21 & 36
        self.place_at_grid(eq_explicit, 'B2', scale_factor=1.2)
        self.place_at_grid(circle_icon, 'B5', scale_factor=0.3)
        self.play(Write(eq_explicit), FadeIn(circle_icon))
        self.lecture[0].set_color(WHITE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Fix for issue 22 & 37
        self.place_at_grid(eq_implicit, 'D2', scale_factor=1.0)
        self.play(Write(eq_implicit))
        self.lecture[1].set_color("#00FFFF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Fix for issue 20 & 35
        circle = Circle(radius=1.5, color=YELLOW)
        self.place_at_grid(circle, 'E5', scale_factor=0.6)
        
        # Highlight variables (Asset integration)
        x_highlight = SurroundingRectangle(eq_explicit[0][4], color=GOLD)
        y_highlight = SurroundingRectangle(eq_implicit[0][2], color=GOLD)
        
        self.play(Create(circle), Create(x_highlight), Create(y_highlight))
        self.lecture[2].set_color(YELLOW)
        self.wait(2)
