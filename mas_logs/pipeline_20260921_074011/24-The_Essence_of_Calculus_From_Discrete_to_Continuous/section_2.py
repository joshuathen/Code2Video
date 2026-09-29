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
        self.setup_layout("The Derivative: Magnifying the Instant", [
            "We zoom in on the cheetah's position curve.", 
            "As we zoom, the curve looks straight.", 
            "This straight line represents instantaneous velocity."
        ])
        
        # Assets
        cheetah = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg")
        
        # Setup Curve - Fixing according to Critic (Issue 24)
        curve = FunctionGraph(lambda x: 0.1 * x**3, x_range=[-3, 3], color=WHITE)
        self.place_in_area(curve, "B3", "F6", scale_factor=1.0)

        # === Animation for Lecture Line 1 ===
        cheetah.scale(0.5).next_to(curve.get_start(), UP)
        self.play(Create(curve), FadeIn(cheetah))
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        
        # === Animation for Lecture Line 2 ===
        # Fixing point placement according to Critic (Issue 25)
        dot = Dot(color="#FFFF00")
        self.place_at_grid(dot, "D4", scale_factor=0.6)
        
        self.add(dot)
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        # Zooming effect
        self.play(curve.animate.scale(2).move_to(dot), run_time=2)
        
        # === Animation for Lecture Line 3 ===
        # Fixing tangent placement according to Critic (Issue 26)
        tangent = Line(start=LEFT*1, end=RIGHT*1, color="#00FFFF")
        self.place_in_area(tangent, "C3", "E5", scale_factor=0.7)
        
        self.play(Create(tangent))
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        
        # Cheetah moves along tangent
        cheetah.move_to(tangent.get_start())
        self.play(MoveAlongPath(cheetah, tangent), run_time=2)
        self.wait(1)
