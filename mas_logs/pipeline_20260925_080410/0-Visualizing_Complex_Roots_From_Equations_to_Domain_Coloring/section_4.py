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

class Section4Scene(Scene):
    def construct(self):
        # Header
        title = Text("Visualization: The Domain Coloring Technique", font_size=32)
        title.to_edge(UP)
        self.add(title)

        # Lecture Text
        lecture_lines = [
            "Domain coloring maps values to colors.",
            "Hue represents angle and brightness shows magnitude.",
            "Dark regions reveal hidden roots instantly."
        ]
        lecture_group = VGroup(*[Text(line, font_size=20) for line in lecture_lines])
        lecture_group.arrange(DOWN, aligned_edge=LEFT)
        lecture_group.to_edge(LEFT, buff=0.5)
        self.add(lecture_group)

        # Objects
        domain = Square(side_length=1.5, fill_opacity=0.5, color=WHITE)
        phase_legend = VGroup(
            Line(ORIGIN, 0.5 * RIGHT, color=RED),
            Line(ORIGIN, 0.5 * RIGHT, color=GREEN),
            Line(ORIGIN, 0.5 * RIGHT, color=BLUE)
        ).arrange(DOWN)

        # Animation Sequence
        # Step 1
        self.play(Create(domain))
        domain.move_to(RIGHT * 1.5 + UP * 0.5)
        self.play(lecture_group[0].animate.set_color(YELLOW))
        self.wait(0.5)

        # Step 2
        phase_legend.move_to(RIGHT * 4.0 + UP * 0.5)
        self.play(FadeIn(phase_legend))
        self.play(lecture_group[1].animate.set_color(GREEN))
        self.wait(0.5)

        # Step 3
        root_spot = Dot(color=MAROON, radius=0.2)
        root_spot.move_to(RIGHT * 1.5 + DOWN * 1.0)
        label = Text("Root", font_size=24)
        label.next_to(root_spot, DOWN)
        
        self.play(FadeIn(root_spot), Write(label))
        self.play(lecture_group[2].animate.set_color(ORANGE))
        self.wait(1)
