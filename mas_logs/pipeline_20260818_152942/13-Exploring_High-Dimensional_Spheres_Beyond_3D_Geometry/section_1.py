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
        lecture_lines = [
            "N-dimensional spaces extend our familiar geometry.",
            "2D circles and 3D spheres serve as foundations.",
            "Distances are calculated via the Euclidean norm.",
            "Higher dimensions define points as hyper-bubbles.",
            "Constraint: sum of squares equals radius squared."
        ]
        self.setup_layout("Introduction: The Concept of Dimension", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg]
        point = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg", color=WHITE)
        self.place_at_grid(point, 'B2', scale_factor=0.8)
        label_0d = Text("0D", font_size=24, color=WHITE)
        self.place_at_grid(label_0d, 'B3', scale_factor=0.7)
        self.play(Create(point), Write(label_0d))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        line = Line(start=LEFT, end=RIGHT, color="#00FF00")
        self.place_at_grid(line, 'C2', scale_factor=0.8)
        label_1d = Text("1D", font_size=24, color="#00FF00")
        self.place_at_grid(label_1d, 'C3', scale_factor=0.7)
        self.play(ReplacementTransform(point, line), ReplacementTransform(label_0d, label_1d))
        self.lecture[1].set_color("#00FF00")

        # === Animation for Lecture Line 3 ===
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg]
        sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color="#0000FF")
        self.place_at_grid(sphere, 'D2', scale_factor=0.8)
        label_2d = Text("2D", font_size=24, color="#0000FF")
        self.place_at_grid(label_2d, 'D3', scale_factor=0.7)
        self.play(ReplacementTransform(line, sphere), ReplacementTransform(label_1d, label_2d))
        self.lecture[2].set_color("#0000FF")
        
        # Remaining lines need color changes
        self.lecture[3].set_color("#FFFF00")
        self.lecture[4].set_color("#FF00FF")
        self.wait(2)
