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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Synthesis and Summary", [
            "Dimensional shifting maps boundaries across N-1 spaces.",
            "Logic helps us visualize what we cannot see.",
            "Mathematical inference defines higher-dimensional interaction."
        ])
        
        # Visual assets
        shapes_2d = VGroup(
            Square(color=BLUE),
            Circle(color=RED),
            RegularPolygon(3, color=YELLOW)
        ).arrange(RIGHT)
        label_1 = Text("Flatland Shapes", font_size=18, color=WHITE)
        
        conn_3d = VGroup(
            Cube(side_length=1.5, fill_opacity=0.2, color=GREEN),
            Sphere(radius=0.75, fill_opacity=0.2, color=ORANGE)
        ).arrange(RIGHT)
        label_2 = Text("3D Connection", font_size=18, color=GREEN)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.place_in_area(shapes_2d, 'A2', 'B5', scale_factor=0.7)
        self.place_at_grid(label_1, 'C2', scale_factor=0.8)
        self.play(Create(shapes_2d), Write(label_1))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        self.place_in_area(conn_3d, 'D2', 'E5', scale_factor=0.7)
        self.place_at_grid(label_2, 'F3', scale_factor=0.8)
        self.play(Create(conn_3d), Write(label_2))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.play(FadeOut(shapes_2d), FadeOut(label_1), FadeOut(conn_3d), FadeOut(label_2))
        self.wait(2)
