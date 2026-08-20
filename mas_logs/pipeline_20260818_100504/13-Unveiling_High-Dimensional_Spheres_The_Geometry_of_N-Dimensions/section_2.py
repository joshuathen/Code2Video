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
        self.setup_layout("Defining the N-Sphere", [
            "An n-sphere is points at constant distance r.", 
            "We define spheres based on their surface manifold.", 
            "A 3-sphere bounds a 4D hypersphere object."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Show point (dim 0, color #FFFFFF).
        point = Dot(color="#FFFFFF")
        self.place_at_grid(point, "B3", scale_factor=0.8)
        self.play(FadeIn(point))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Draw line (dim 1, color #FF5733).
        line = Line(start=self.grid["B3"], end=self.grid["B5"], color="#FF5733")
        self.play(Create(line))
        self.lecture[1].set_color("#FF5733")

        # === Animation for Lecture Line 3 ===
        # Draw circle (dim 2, color #33FF57).
        circle = Circle(radius=0.5, color="#33FF57")
        self.place_in_area(circle, "D2", "E3", scale_factor=1.2)
        circle_label = Text("2-Sphere", font_size=18, color="#33FF57")
        self.place_in_area(circle_label, "D3", "D3", scale_factor=0.7)
        self.play(Create(circle), Write(circle_label))
        self.lecture[2].set_color("#33FF57")
        
        self.wait(2)
