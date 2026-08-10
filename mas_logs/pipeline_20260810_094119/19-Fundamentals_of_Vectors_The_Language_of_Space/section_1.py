from manim import *
import os

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
        self.setup_layout("Introduction: Scalars vs. Vectors", [
            "Scalars hold only magnitude information.",
            "Vectors describe both magnitude and direction.",
            "Imagine a cat weighing five kilograms.",
            "Its jumping velocity is a vector.",
            "Vectors are arrows in our space."
        ])
        
        # === Animation for Lecture Line 1 ===
        s = Dot(color="#FFD700")
        s_label = Text("Scalar", font_size=20, color="#FFD700")
        cat_img = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png")
        
        # Asset usage: cat next to scalar
        cat_img.scale(0.2)
        cat_img.next_to(s, LEFT, buff=0.2)
        
        self.place_at_grid(s, 'B2', scale_factor=1.0)
        self.place_at_grid(s_label, 'C2', scale_factor=0.8)
        self.play(FadeIn(s), FadeIn(s_label), FadeIn(cat_img))
        self.lecture[0].set_color("#FFD700")

        # === Animation for Lecture Line 2 ===
        v = Arrow(start=ORIGIN, end=RIGHT*1.5, color="#00CED1")
        v_label = Text("Vector", font_size=20, color="#00CED1")
        
        # Using centered placement per feedback
        self.place_at_grid(v, 'B5', scale_factor=1.0)
        self.place_at_grid(v_label, 'C5', scale_factor=0.8)
        self.play(FadeIn(v), FadeIn(v_label))
        self.lecture[1].set_color("#00CED1")

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFD700")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00CED1")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#00CED1")
        self.wait(1)
