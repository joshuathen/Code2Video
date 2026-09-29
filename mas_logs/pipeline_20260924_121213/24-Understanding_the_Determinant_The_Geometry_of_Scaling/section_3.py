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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Negative determinant means a flip.", "Zero determinant squashes the plane.", "Dimension is lost at zero."]
        self.setup_layout("The Meaning of Sign and Zero", lecture_lines)
        
        # Setup visual elements using SVGMobject asset
        plane = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plane.svg")
        self.place_at_grid(plane, "B5", scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        # Negative determinant means a flip.
        self.lecture[0].set_color(YELLOW)
        # Flip visual: reflection of plane
        plane_flipped = plane.copy().set_color(RED).flip(axis=UP)
        self.play(Transform(plane, plane_flipped), run_time=1.5)

        # === Animation for Lecture Line 2 ===
        # Zero determinant squashes the plane.
        self.lecture[1].set_color("#FF00FF")
        # Squashing: flatten to line
        line = Line(start=LEFT*1, end=RIGHT*1, color="#FF00FF", stroke_width=6)
        self.place_at_grid(line, "D5", scale_factor=0.8)
        # Flatten plane to line
        self.play(plane.animate.stretch(0, 1), run_time=1.5)
        self.play(FadeIn(line))

        # === Animation for Lecture Line 3 ===
        # Dimension is lost at zero.
        self.lecture[2].set_color(RED)
        lost_label = Text("Lost Dimension", color=RED, font_size=20)
        self.place_at_grid(lost_label, "E5", scale_factor=0.8)
        self.play(Write(lost_label), Indicate(line))
        self.wait(1)
