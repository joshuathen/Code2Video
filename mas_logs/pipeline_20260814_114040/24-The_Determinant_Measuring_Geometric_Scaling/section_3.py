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
        self.setup_layout("Geometric Signification: Orientation and Zero", [
            "Negative determinants indicate a mirror-image flip.",
            "A zero determinant collapses shapes into lines.",
            "Zero means the transformation loses a dimension."
        ])
        
        # Helper for shapes
        def get_parallelogram(color=GREEN):
            return Polygon(ORIGIN, RIGHT*1.5+UP*0.5, RIGHT*2.5+UP*1.5, RIGHT*1.0+UP*1.0, color=color, fill_opacity=0.3)

        # === Animation for Lecture Line 1 ===
        shape1 = get_parallelogram(GREEN)
        self.place_at_grid(shape1, 'C2', scale_factor=0.8)
        self.play(Create(shape1))
        self.lecture[0].set_color(GREEN)
        
        # Mirror asset
        mirror = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mirror.svg")
        self.place_at_grid(mirror, 'C4', scale_factor=0.5)
        self.play(FadeIn(mirror))
        
        # Reflection
        shape_flipped = get_parallelogram(RED)
        self.place_at_grid(shape_flipped, 'C5', scale_factor=0.8)
        shape_flipped.flip(UP)
        self.play(TransformFromCopy(shape1, shape_flipped))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        line_vec = Line(ORIGIN, RIGHT*2, color=YELLOW)
        self.place_at_grid(line_vec, 'E2', scale_factor=0.9)
        self.play(FadeOut(shape1), FadeOut(shape_flipped), FadeOut(mirror))
        self.play(Create(line_vec))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(BLUE)
        point = Dot(color=BLUE)
        self.place_at_grid(point, 'E5', scale_factor=0.6)
        self.play(Transform(line_vec, point))
        self.wait(1)
