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
        self.setup_layout("The 'Why': Explicit vs. Implicit", [
            "We know functions like y equals x squared.",
            "Some curves, like circles, trap y inside.",
            "Implicit differentiation lets us find their slope."
        ])
        
        # Animation Elements
        explicit_text = Text("y = x^2", color=WHITE)
        implicit_text = Text("x^2 + y^2 = r^2", color="#FFD700")
        diff_text = Text("Implicit differentiation finds slopes.", color=WHITE)
        question_mark = Text("?", font_size=72, color=WHITE)
        
        # Asset
        circle_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg")

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        # Adjust layout based on feedback: B3
        self.place_at_grid(explicit_text, 'B3', scale_factor=0.6)
        self.play(Write(explicit_text))
        self.play(explicit_text.animate.set_color("#00FF00"))
        self.play(self.lecture[0].animate.set_color("#00FF00"))

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        # Adjust layout based on feedback: C4
        self.place_at_grid(implicit_text, 'C4', scale_factor=0.6)
        # Position circle icon near formula
        self.place_at_grid(circle_icon, 'C3', scale_factor=0.4)
        self.play(Write(implicit_text), FadeIn(circle_icon))
        self.play(implicit_text.animate.set_color("#FFD700"))
        self.play(self.lecture[1].animate.set_color("#FFD700"))

        # === Animation for Lecture Line 3 ===
        self.play(FadeOut(explicit_text), FadeOut(implicit_text), FadeOut(circle_icon))
        self.play(FadeIn(self.lecture[2]))
        # Adjust layout based on feedback: B4 for diff_text, C5 for question_mark
        self.place_at_grid(diff_text, 'B4', scale_factor=0.6)
        self.place_at_grid(question_mark, 'C5', scale_factor=0.8)
        self.play(Write(diff_text), Write(question_mark))
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        self.wait(1)
