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
        self.setup_layout("Introduction & Prerequisites", [
            "A circle has an area of pi r squared.",
            "A sphere is a three-dimensional curved surface.",
            "Parallel light rays create a circular shadow."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display title 'Introduction' in #FFFFFF, scale up.
        self.play(self.lecture[0].animate.set_color("#00FFFF"))

        # === Animation for Lecture Line 2 ===
        # Fade in [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg] labeled 'S' in #FF00FF.
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        # Using SVGMobject for asset
        sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color="#FF00FF")
        label_s = Text("S", font_size=24, color="#FF00FF")
        self.place_at_grid(sphere, "C2", scale_factor=1.2)
        self.place_at_grid(label_s, "C5", scale_factor=0.8)
        self.play(FadeIn(sphere), Write(label_s))

        # === Animation for Lecture Line 3 ===
        # Highlight the sphere center 'O' in #FFFF00.
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        dot_o = Dot(color="#FFFF00")
        label_o = Text("O", font_size=20, color="#FFFF00")
        self.place_at_grid(dot_o, "D2", scale_factor=0.5)
        self.place_at_grid(label_o, "E2", scale_factor=0.8)
        self.play(GrowFromCenter(dot_o), Write(label_o))
        self.wait(2)
