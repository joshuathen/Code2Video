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
        lecture_lines = ["Hilbert curves are a more efficient alternative.", "The L-shape logic preserves spatial locality.", "Points close on line remain close in space."]
        self.setup_layout("The Hilbert Curve: Efficiency in Recursion", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        # Simple visual representing \"efficiency\": a growing curve
        hilbert_base = VGroup(
            Line(UP, ORIGIN), Line(ORIGIN, RIGHT), Line(RIGHT, UP+RIGHT)
        ).set_color(YELLOW)
        self.place_in_area(hilbert_base, 'A4', 'C6', scale_factor=0.6)
        self.play(Create(hilbert_base))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        # Visualization for L-shape rotation (mimicking [Asset: Hilbert_L_Shape_Evolution])
        l_shape = VGroup(
            Line(UP, ORIGIN), Line(ORIGIN, RIGHT)
        ).set_color(RED)
        self.place_at_grid(l_shape, 'D5', scale_factor=0.8)
        self.play(Create(l_shape))
        self.play(l_shape.animate.rotate(PI/2, about_point=self.grid['D5']))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(ORANGE))
        # Points to show locality
        p1 = Dot(color=PURPLE)
        p2 = Dot(color=PURPLE)
        self.place_at_grid(p1, 'D2', scale_factor=0.9)
        self.place_at_grid(p2, 'D3', scale_factor=0.9)
        self.play(Create(p1), Create(p2))
        self.play(Flash(p1), Flash(p2))

        self.wait(2)
