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
        self.setup_layout("Geometric Interpretation of Linear Systems", [
            "Systems represent intersections of geometric objects.",
            "Solutions are the specific meeting points.",
            "Matrices act as linear transformations.",
            "Visualizing movement clarifies algebraic results.",
            "Linear algebra links geometry and computation."
        ])
        
        # Setup Axes
        axes = Axes(x_range=[-1, 5, 1], y_range=[-1, 4, 1], x_length=5, y_length=4)
        self.place_in_area(axes, 'B3', 'E4', scale_factor=0.65)
        self.add(axes)

        # === Animation for Lecture Line 1 ===
        # Draw a 2D plane with lines #FF5733 (Line A) and #33FF57 (Line B).
        line_a = axes.plot(lambda x: 5 - 2*x, color="#FF5733")
        line_b = axes.plot(lambda x: x - 1, color="#33FF57")
        self.play(Create(line_a), Create(line_b), self.lecture[0].animate.set_color("#FF5733"))

        # === Animation for Lecture Line 2 ===
        # Highlight the intersection point #FFFF33
        intersection = Dot(axes.c2p(2, 1), color="#FFFF33", radius=0.1)
        intersection_label = Text("(2, 1)", font_size=20, color="#FFFF33")
        self.place_at_grid(intersection_label, 'D4', scale_factor=0.7)
        self.play(FadeIn(intersection), FadeIn(intersection_label), self.lecture[1].animate.set_color("#FFFF33"))

        # === Animation for Lecture Line 3 ===
        # Animate a vector #33A1FF moving from origin to intersection point.
        vector = Vector(axes.c2p(2, 1), color="#33A1FF")
        self.play(GrowArrow(vector), self.lecture[2].animate.set_color("#33A1FF"))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(WHITE))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
