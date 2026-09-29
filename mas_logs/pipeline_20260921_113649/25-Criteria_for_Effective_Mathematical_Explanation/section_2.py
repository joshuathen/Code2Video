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
        self.setup_layout("Criterion 1: Scaffolding and Prerequisite Alignment", [
            "Activate prior knowledge before complex concepts.",
            "Use visual layering to build foundations.",
            "Example: Nested gears visualize function composition."
        ])

        scaffolding = Text("Scaffolding", color=WHITE)
        prereq = Text("Prerequisites", color=WHITE)
        gears = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gears.svg")
        
        # Apply positioning constraints
        self.place_at_grid(scaffolding, 'B4', scale_factor=0.7)
        self.place_at_grid(prereq, 'C4', scale_factor=0.7)
        self.place_at_grid(gears, 'D4', scale_factor=0.5)
        
        line = Line(scaffolding.get_bottom(), prereq.get_top(), color=WHITE)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.play(FadeIn(scaffolding), FadeIn(gears))

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        self.play(Create(line))

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        self.play(prereq.animate.set_color("#FF0000"))
        self.play(Flash(prereq, color="#FF0000", line_length=0.2))
        self.wait(1)
