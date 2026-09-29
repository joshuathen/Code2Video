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
        self.setup_layout("Graphical Synthesis & Comparison", [
            "Compare radial expansion and circular rotation.",
            "Field A shows divergence without curl.",
            "Field B shows curl without divergence."
        ])
        
        # Animation Elements
        def radial_field(p):
            x, y = p[0], p[1]
            return np.array([x, y, 0]) * 0.2

        def circular_field(p):
            x, y = -p[1], p[0]
            return np.array([x, y, 0]) * 0.2

        field_a = StreamLines(radial_field, stroke_width=2, color=BLUE, x_range=[-1.5, 1.5], y_range=[-1.5, 1.5])
        field_b = StreamLines(circular_field, stroke_width=2, color=RED, x_range=[-1.5, 1.5], y_range=[-1.5, 1.5])
        
        # Labels
        label_a = Text("Field A (Divergence)", font_size=20, color=BLUE)
        label_b = Text("Field B (Curl)", font_size=20, color=RED)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.place_at_grid(field_a, "B2", scale_factor=0.5)
        self.place_at_grid(field_b, "B5", scale_factor=0.5)
        self.place_at_grid(label_a, "C2", scale_factor=0.8)
        self.place_at_grid(label_b, "C5", scale_factor=0.8)
        self.play(Create(field_a), Create(field_b), Write(label_a), Write(label_b))
        self.play(field_a.animate.start_animation(), field_b.animate.start_animation())

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        highlight_a = Circle(radius=0.5, color="#FF00FF").move_to(field_a.get_center())
        self.play(Create(highlight_a))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        highlight_b = Circle(radius=0.5, color="#FF00FF").move_to(field_b.get_center())
        self.play(Transform(highlight_a, highlight_b))
        self.play(FadeOut(highlight_a))
        self.wait(2)
