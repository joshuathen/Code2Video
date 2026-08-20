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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Null space contains vectors squashed to origin.",
            "This hidden part disappears during transformation.",
            "It represents dimensions lost in projection."
        ]
        self.setup_layout("Null Space: The Hidden Dimensions", lecture_lines)
        
        # Elements
        origin = Dot(self.grid["E3"], color=WHITE)
        origin_label = Text("Origin", font_size=18).next_to(origin, DOWN)
        
        # Using asset as per instruction [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg]
        # Since it is '/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg', we use a proxy shape that acts as the required icon.
        icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg") if os.path.exists("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg") else Circle(radius=0.2, color=WHITE)
        
        transformation_vectors = VGroup(
            Arrow(start=self.grid["B2"], end=self.grid["E3"], color=BLUE),
            Arrow(start=self.grid["B4"], end=self.grid["E3"], color=BLUE),
            icon.copy()
        )
        self.place_in_area(transformation_vectors, "B4", "D6", scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(origin, origin_label), FadeIn(transformation_vectors))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF0000")
        vec_null = Arrow(start=self.grid["C5"], end=self.grid["D5"], color=RED)
        null_vector_label = Text("Null Vector", color=RED, font_size=16)
        self.place_at_grid(null_vector_label, "F5", scale_factor=0.7)
        self.play(Create(vec_null), Write(null_vector_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF0000")
        null_space_area = Rectangle(width=2, height=0.5, color=RED, fill_opacity=0.3)
        self.place_in_area(null_space_area, "D3", "E5", scale_factor=0.6)
        label_space = Text("Null Space", color=RED, font_size=18).next_to(null_space_area, UP)
        
        # Using icon asset again
        icon2 = icon.copy()
        self.place_at_grid(icon2, "C4", scale_factor=0.5)
        
        self.play(FadeIn(null_space_area), Write(label_space), FadeIn(icon2))
        self.wait(2)
