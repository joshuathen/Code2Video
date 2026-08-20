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
        self.setup_layout("Synthesis and Visualization", [
            "Changing basis shifts our perspective.",
            "The physical object remains constant.",
            "Coordinates are just numerical representations."
        ])
        
        # Grid setup
        grid = NumberPlane(x_range=[-3, 3], y_range=[-3, 3], axis_config={"include_numbers": False})
        self.place_in_area(grid, "C3", "F6", scale_factor=0.5)
        self.add(grid)
        
        # Point P (Asset integration)
        p = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/object.svg")
        self.place_at_grid(p, "C5", scale_factor=0.7)
        label_p = Text("P", color=YELLOW, font_size=20).next_to(p, UR, buff=0.1)
        
        # Basis vectors
        b1 = Vector(RIGHT, color=BLUE)
        b2 = Vector(UP, color=RED)
        b1.shift(grid.get_center())
        b2.shift(grid.get_center())
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(Create(b1), Create(b2), FadeIn(p), Write(label_p))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(RED))
        
        # Transform basis to show perspective shift
        new_b1 = Vector(0.5*RIGHT + 0.8*UP, color=BLUE)
        new_b2 = Vector(-0.8*RIGHT + 0.5*UP, color=RED)
        new_b1.shift(grid.get_center())
        new_b2.shift(grid.get_center())
        
        grid_new = grid.copy()
        grid_new.apply_matrix([[0.5, -0.8], [0.8, 0.5]])
        
        self.play(
            ReplacementTransform(grid, grid_new),
            ReplacementTransform(b1, new_b1),
            ReplacementTransform(b2, new_b2)
        )
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        self.wait(2)
