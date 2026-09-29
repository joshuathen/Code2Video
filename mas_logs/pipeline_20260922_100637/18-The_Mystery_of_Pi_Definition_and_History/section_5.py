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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary and Conclusion", [
            "Pi bridges linear and circular geometry.", 
            "It is a universal, natural constant.", 
            "Pi connects science, math, and technology."
        ])
        
        # Elements
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")
        circle = Circle(radius=0.5, color=BLUE)
        line = Line(start=LEFT*0.5, end=RIGHT*0.5, color=RED)
        pi_symbol = MathTex(r"\\pi", font_size=96, color=YELLOW)
        dot = Dot(color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        # Positioned circle/line/compass at A5 as requested by Critic
        compass_group = VGroup(circle, line, compass).arrange(RIGHT)
        self.place_at_grid(compass_group, 'A5', scale_factor=0.6)
        self.play(FadeIn(compass_group), self.lecture[0].animate.set_color(BLUE))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Positioned dot at D5 as requested by Critic
        self.place_at_grid(dot, 'D5', scale_factor=1)
        self.play(FadeIn(dot), self.lecture[1].animate.set_color(GREEN))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Positioned pi_symbol/protractor at F5 as requested by Critic
        math_group = VGroup(pi_symbol, protractor).arrange(RIGHT)
        self.place_at_grid(math_group, 'F5', scale_factor=0.8)
        self.play(FadeIn(math_group), self.lecture[2].animate.set_color(YELLOW))
        self.wait(2)
