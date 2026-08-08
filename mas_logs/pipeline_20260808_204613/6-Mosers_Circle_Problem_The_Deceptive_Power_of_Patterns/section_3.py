from manim import *
import numpy as np

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
        lecture_lines = ["Let's check six points.", "Expect 32? Not quite.", "We only find 31 regions.", "The pattern fails here.", "Counting shows the truth."]
        self.setup_layout("The Reality Check (n=6)", lecture_lines)
        
        # Define assets
        pencil = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pencil.svg", color=WHITE)
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg", color=WHITE)
        
        # Define objects
        circle = Circle(radius=1.2, color=BLUE)
        self.place_in_area(circle, 'B3', 'E5', scale_factor=0.9)
        
        # Define 6 points on the circle
        points = VGroup(*[Dot(circle.point_from_proportion(i/6)) for i in range(6)])
        
        # Prepare text elements
        count_label = Text("Regions: ?", font_size=32, color=YELLOW)
        self.place_at_grid(count_label, 'C4', scale_factor=1.0)
        
        expect_label = Text("Expected: 32", font_size=24, color=GRAY)
        self.place_at_grid(expect_label, 'F4', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.place_at_grid(pencil, 'B2', scale_factor=0.5)
        self.play(Create(circle), FadeIn(pencil))
        self.play(FadeIn(points))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(FadeIn(expect_label))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(RED)
        actual_count = Text("Regions: 31", font_size=32, color=RED)
        actual_count.move_to(count_label.get_center())
        self.play(Transform(count_label, actual_count))
        
        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        self.play(Indicate(count_label))

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(GREEN)
        self.place_at_grid(protractor, 'E2', scale_factor=0.6)
        self.play(FadeIn(protractor))
        self.play(FadeOut(expect_label), FadeOut(pencil))
        self.wait(1)
