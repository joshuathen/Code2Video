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
        self.setup_layout("The Critical Strip and the Hypothesis", [
            "The critical strip lies between 0 and 1.",
            "The Riemann Hypothesis targets the critical line.",
            "It predicts where all non-trivial zeros land."
        ])
        
        # Grid area for complex plane: A2 to C5
        plane = Axes(
            x_range=[-1, 2, 1],
            y_range=[-2, 2, 1],
            axis_config={"include_tip": False}
        ).scale(0.4)
        
        # Apply layout constraints requested by Critic
        self.place_in_area(plane, 'A2', 'C5', scale_factor=0.6)
        self.add(plane)
        
        # === Animation for Lecture Line 1 ===
        # The critical strip lies between 0 and 1.
        self.lecture[0].set_color("#E6E6FA")
        strip = Rectangle(
            height=plane.get_y_axis().length,
            width=plane.c2p(1, 0)[0] - plane.c2p(0, 0)[0],
            fill_color="#E6E6FA",
            fill_opacity=0.3,
            stroke_width=0
        )
        strip.move_to(plane.c2p(0.5, 0))
        self.play(FadeIn(strip))
        
        # === Animation for Lecture Line 2 ===
        # The Riemann Hypothesis targets the critical line.
        self.lecture[1].set_color("#FFFFFF")
        critical_line = Line(
            plane.c2p(0.5, -1.8),
            plane.c2p(0.5, 1.8),
            color=WHITE,
            stroke_width=4
        )
        self.play(Create(critical_line))
        
        # === Animation for Lecture Line 3 ===
        # It predicts where all non-trivial zeros land.
        self.lecture[2].set_color("#FF0000")
        zeros = VGroup(
            Dot(plane.c2p(0.5, 1.2), color=RED),
            Dot(plane.c2p(0.5, -1.2), color=RED),
            Dot(plane.c2p(0.5, 0.5), color=RED),
            Dot(plane.c2p(0.5, -0.5), color=RED)
        )
        self.play(FadeIn(zeros))
        self.wait(1)
