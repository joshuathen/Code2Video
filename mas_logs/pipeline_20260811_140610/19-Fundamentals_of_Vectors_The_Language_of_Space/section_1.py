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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines_text = ["Vectors possess both magnitude and direction.", "Scalars are represented by single numbers.", "Think of vectors as directed paths."]
        self.setup_layout("Introduction: What is a Vector?", lecture_lines_text)
        
        # Initialize objects
        origin = Dot(color=WHITE)
        target_point = Dot(color=WHITE)
        self.place_at_grid(origin, 'B3', scale_factor=0.6)
        self.place_at_grid(target_point, 'D5', scale_factor=0.6)
        
        vector = Arrow(start=origin.get_center(), end=target_point.get_center(), color="#FF5733", buff=0)
        label_v = Text("Vector V", font_size=24, color="#FF5733").scale(0.7)
        label_v.next_to(target_point, UP, buff=0.1)
        vector_group = VGroup(vector, label_v)
        
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.play(Create(origin), Create(target_point))
        self.place_at_grid(vector_group, 'C4', scale_factor=1.0)
        self.play(Create(vector), Write(label_v))
        self.lecture[0].set_color("#FF5733")

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        self.place_at_grid(ruler, 'E4', scale_factor=0.5)
        self.play(FadeIn(ruler))
        self.play(vector.animate.scale(1.1).set_color("#FFFF00"), run_time=0.5)
        self.play(vector.animate.scale(1/1.1).set_color("#FF5733"), run_time=0.5)
        self.play(FadeOut(ruler))
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        self.place_at_grid(compass, 'D2', scale_factor=0.5)
        self.play(FadeIn(compass))
        h_origin = Circle(radius=0.2, color="#FFFF00").move_to(origin)
        h_point = Circle(radius=0.2, color="#FFFF00").move_to(target_point)
        self.play(Create(h_origin), Create(h_point))
        self.lecture[2].set_color("#FFFF00")
        self.wait(2)
