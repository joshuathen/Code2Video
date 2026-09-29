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
        lecture_lines = [
            "A 3D shape passing through a 2D plane.",
            "The plane only sees a changing 2D cross-section.",
            "Reconstruct the full 3D shape from these slices.",
            "Dimensional shifting reveals hidden patterns in data.",
            "See the whole by studying its individual slices."
        ]
        self.setup_layout("Application: The Dimensional Traveler", lecture_lines)
        
        plane = Rectangle(width=4, height=4, color=WHITE, fill_opacity=0.2)
        self.place_in_area(plane, "B1", "E6")
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg
        sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color=WHITE)
        self.place_at_grid(sphere, "D4", scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        sphere.shift(UP * 2)
        self.play(FadeIn(sphere), FadeIn(plane))
        self.play(sphere.animate.shift(DOWN * 4), run_time=2)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF0000")
        circle = Circle(radius=0.5, color="#FF0000", fill_opacity=0.8)
        self.place_at_grid(circle, "D2", scale_factor=0.5)
        self.play(Create(circle))
        self.play(circle.animate.scale(2), run_time=1)
        self.play(circle.animate.scale(0.5), run_time=1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#0000FF")
        circles = VGroup(*[Circle(radius=r, color="#0000FF", fill_opacity=0.5).move_to(self.grid["C3"]) for r in [0.2, 0.5, 0.8, 0.5, 0.2]])
        self.play(FadeOut(circle), FadeIn(circles.arrange(RIGHT, buff=0.1).scale(0.5).move_to(self.grid["D4"])))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFF00")
        path = DashedLine(start=self.grid["B2"], end=self.grid["E5"], color="#FFFF00")
        self.play(Create(path))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFFFF")
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg
        full_sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color=WHITE)
        self.place_at_grid(full_sphere, "E5", scale_factor=0.7)
        self.play(FadeOut(circles), FadeOut(path), FadeIn(full_sphere))
        self.wait(1)
