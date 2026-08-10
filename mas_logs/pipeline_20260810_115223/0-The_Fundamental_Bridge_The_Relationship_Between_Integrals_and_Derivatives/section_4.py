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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Differentiating an integral recovers the original function.", "The process effectively undoes the integration.", "It is a perfect mathematical cancellation."]
        self.setup_layout("The Core Connection: Cancellation", lecture_lines)
        
        # Define mobjects once
        f_x = MathTex("f(x)", color=BLUE)
        integral = MathTex("\\int f(t) dt", color=YELLOW)
        derivative = MathTex("\\frac{d}{dx} \\int f(t) dt", color=RED)
        equals = MathTex("=", color=WHITE)
        result = MathTex("f(x)", color=GREEN)
        
        # Assets
        eraser = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/eraser.svg").scale(0.5)
        scissors = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scissors.svg").scale(0.5)
        
        gap_vis = VGroup(f_x, integral).arrange(RIGHT, buff=0.5)
        self.place_at_grid(gap_vis, 'B5', scale_factor=0.9) # Fix for Issue 30
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.place_in_area(derivative, 'C4', 'D5', scale_factor=0.8) # Fix for Issue 29
        self.place_at_grid(eraser, 'D2', scale_factor=0.8) # Integration of Asset
        self.play(FadeIn(derivative), FadeIn(eraser))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(FadeIn(equals), FadeIn(result))
        self.place_at_grid(VGroup(equals, result), 'D5', scale_factor=1.0)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#AA33FF")
        
        # Highlighting the cancellation
        self.place_at_grid(scissors, 'E2', scale_factor=0.8) # Integration of Asset
        box = SurroundingRectangle(VGroup(derivative, equals, result), color="#AA33FF")
        self.play(Create(box), FadeIn(scissors))
        self.play(Indicate(box))
        self.wait(2)
