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
        self.setup_layout("Problem Definition: The Windmill Motion", [
            "Consider points distributed in a plane.",
            "A line pivots around one point.",
            "The line sweeps until it hits another."
        ])

        # Assets
        points = VGroup(*[Dot(color=WHITE) for _ in range(5)])
        for i, p in enumerate(points):
            self.place_at_grid(p, f"{['B1', 'C2', 'D3', 'E4', 'F5'][i]}")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.play(FadeIn(points))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        pivot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/windmill.svg")
        self.place_at_grid(pivot, 'D3', scale_factor=0.6)
        line = Line(start=pivot.get_center(), end=pivot.get_center() + RIGHT * 2, color="#FFFF00")
        self.play(FadeIn(pivot), Create(line))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        target = Dot(color="#00FF00")
        self.place_at_grid(target, 'D5', scale_factor=0.6)
        
        windmill_system = VGroup(pivot, line)
        self.place_in_area(windmill_system, 'D3', 'D5', scale_factor=0.7)
        
        self.play(
            Rotate(line, angle=PI/4, about_point=pivot.get_center()),
            FadeIn(target)
        )
