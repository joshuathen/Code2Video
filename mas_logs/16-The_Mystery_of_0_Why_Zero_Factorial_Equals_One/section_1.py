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
        # Initial Setup
        title = "The Counter-Intuitive Question"
        lines = [
            "Meet Professor Pi, here to solve a mathematical mystery.",
            "We know factorials like three, two, and one.",
            "But what happens when we reach zero factorial?"
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # [Asset: Professor Pi] (Owl placeholder)
        # Creating a simple OWL representation using basic shapes
        owl_body = Ellipse(width=1.2, height=1.5, color=WHITE, fill_opacity=1)
        owl_eye_l = Circle(radius=0.15, color=BLACK, fill_opacity=1).shift(LEFT*0.25 + UP*0.3)
        owl_eye_r = Circle(radius=0.15, color=BLACK, fill_opacity=1).shift(RIGHT*0.25 + UP*0.3)
        owl_beak = Triangle(color=ORANGE, fill_opacity=1).scale(0.1).rotate(PI).shift(UP*0.1)
        professor_pi = VGroup(owl_body, owl_eye_l, owl_eye_r, owl_beak)
        pi_label = Text("Professor Pi", font_size=18, color=WHITE).next_to(owl_body, DOWN, buff=0.1)
        owl_asset = VGroup(professor_pi, pi_label)
        
        # Grid positioning: Center-left within the animation area
        self.place_in_area(owl_asset, "B1", "E2", scale_factor=1.0)
        
        # Animation: Slide in from the left
        owl_asset.shift(LEFT * 6)
        self.play(owl_asset.animate.shift(RIGHT * 6), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Chalkboard
        chalkboard = Rectangle(width=3.5, height=4.5, fill_color="#A52A2A", fill_opacity=1, stroke_color=WHITE)
        self.place_in_area(chalkboard, "A4", "F6")
        
        # Factorial list
        f3 = Text("3! = 6", font_size=32, color=WHITE)
        f2 = Text("2! = 2", font_size=32, color=WHITE)
        f1 = Text("1! = 1", font_size=32, color=WHITE)
        
        factorial_list = VGroup(f3, f2, f1).arrange(DOWN, buff=0.5)
        factorial_list.move_to(chalkboard.get_center() + UP * 0.8)
        
        self.play(Create(chalkboard))
        for item in factorial_list:
            self.play(FadeIn(item, shift=UP*0.2), run_time=0.8)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # 0! and Question Mark
        f0_base = Text("0! = ", font_size=32, color=WHITE)
        f0_base.next_to(factorial_list, DOWN, buff=0.5)
        
        question_mark = Text("?", font_size=60, color="#FFFF00")
        question_mark.next_to(f0_base, RIGHT, buff=0.2)
        
        # Glowing effect for the question mark
        glow = question_mark.copy().set_stroke(width=10, opacity=0.5).set_color("#FFFF00")
        
        self.play(Write(f0_base))
        self.play(
            FadeIn(question_mark),
            FadeIn(glow),
            question_mark.animate.scale(1.2),
            run_time=1
        )
        self.play(Indicate(question_mark, color="#FFFF00"))
        
        self.wait(2)
