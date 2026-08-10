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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisites: Potential and Kinetic Energy", 
                          ["Objects gain speed by dropping quickly.", 
                           "Higher drops convert potential energy to velocity.", 
                           "Faster early speed reduces total travel time."])
        
        # Assets
        ball = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg")
        ramp = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ramp.svg")
        
        # Define Mobjects
        u_formula = MathTex("U = mgh", color=WHITE)
        k_formula = MathTex("K = \\frac{1}{2}mv^2", color=WHITE)
        cons_formula = MathTex("U + K = \\text{constant}", color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(ramp, 'B4', scale_factor=0.6)
        self.place_at_grid(ball, 'B4', scale_factor=0.3)
        self.place_at_grid(u_formula, 'B3', scale_factor=0.9)
        self.play(FadeIn(ramp), FadeIn(ball), FadeIn(u_formula))
        self.lecture[0].set_color("#FF6666")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(k_formula, 'C3', scale_factor=0.9)
        self.play(FadeIn(k_formula), ball.animate.move_to(self.grid['C4']))
        self.lecture[1].set_color("#66FF66")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(cons_formula, 'D3', scale_factor=1.0)
        self.play(Write(cons_formula))
        self.play(cons_formula.animate.set_color("#FFFF00"))
        self.lecture[2].set_color("#FFFF00")
        self.wait(2)
