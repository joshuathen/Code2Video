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
        lecture_lines = ["Pi is more than 3.14.", "It is a limit of processes.", "Observe a polygon approaching a circle."]
        self.setup_layout("Introduction: The Geometric Curiosity", lecture_lines)
        
        # Assets
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg", color=WHITE)
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg", color=WHITE)
        
        circle = Circle(radius=1.0, color=WHITE)
        label_c = Text("C", color=WHITE).next_to(circle, UP)
        label_r = MathTex("r", color=WHITE).next_to(circle, RIGHT)
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(compass, "B2", scale_factor=0.5)
        self.play(FadeIn(compass), FadeIn(circle), Write(label_c), Write(label_r))
        self.lecture[0].set_color("#FFFFFF")
        
        # === Animation for Lecture Line 2 ===
        circumference = MathTex("2 \\pi r", color="#FF0000")
        self.place_in_area(circumference, "B4", "B6", scale_factor=0.9)
        self.play(Write(circumference))
        self.lecture[1].set_color("#FF0000")
        
        # === Animation for Lecture Line 3 ===
        ratio = MathTex("\\frac{\\text{Circumference}}{2r} = \\pi", color="#00FF00")
        self.place_in_area(ratio, "C4", "C6", scale_factor=0.9)
        self.play(Write(ratio))
        self.lecture[2].set_color("#00FF00")
        
        # Final focus
        pi_final = MathTex("\\pi", color="#FFFF00", font_size=72)
        self.place_at_grid(protractor, "C2", scale_factor=0.6)
        self.place_at_grid(pi_final, "D4", scale_factor=1.2)
        self.play(
            FadeOut(circle), FadeOut(label_c), FadeOut(label_r),
            FadeOut(circumference), FadeOut(ratio), FadeOut(compass),
            FadeIn(protractor),
            FadeIn(pi_final)
        )
        self.wait(2)
