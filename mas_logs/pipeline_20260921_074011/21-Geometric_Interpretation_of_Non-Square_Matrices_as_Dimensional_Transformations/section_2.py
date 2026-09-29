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
        self.setup_layout("Expanding Dimensions (m > n)", [
            "Matrices can expand lower dimensions into higher ones.",
            "Imagine a 2D plane inside 3D space.",
            "The matrix defines the plane's position."
        ])
        
        # Define objects
        # 2D plane in 3D using Asset
        plane = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plane.svg", color=GRAY, fill_opacity=0.3)
        self.place_at_grid(plane, 'D4', scale_factor=0.9)
        
        # Projector using Asset
        projection_vector = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plane.svg", color="#FF33A1")
        self.place_at_grid(projection_vector, 'B4', scale_factor=0.8)
        
        # Highlight vector
        highlight_vec = Arrow(start=LEFT*1, end=RIGHT*1, color="#A1FF33", buff=0)
        self.place_in_area(highlight_vec, 'C2', 'C3', scale_factor=0.7)
        
        # Residual vector
        residual_vec = Line(start=UP*0.5, end=DOWN*0.5, color="#33A1FF")
        self.place_at_grid(residual_vec, 'E4', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF33A1"), Create(projection_vector))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#A1FF33"), Create(plane), Create(highlight_vec))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#33A1FF"), Create(residual_vec))
        self.wait(1)
